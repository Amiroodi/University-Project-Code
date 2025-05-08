"""
Contains functions for training and testing a PyTorch model.
"""
import torch
from torch import nn
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from torchmetrics.classification import MulticlassF1Score
# from torch.cuda.amp import GradScaler
# from torch import autocast

scalar = torch.amp.GradScaler('cuda', enabled=True)

def ordinal_labels(y, num_classes):
    """Convert labels to cumulative one-hot encoding"""
    y_cumulative = torch.zeros(len(y), num_classes)
    for i in range(num_classes):
        y_cumulative[:, i] = (y >= i).float()
    return y_cumulative

def reg_classify(x, device):
    bins = torch.tensor([0.5, 1.5, 2.5, 3.5]).to(device)  # Class boundaries
    # Classify using bucketize
    classified = torch.bucketize(x, bins, right=False)  # right=False ensures correct bin placement
    return classified

weight_const = 0.2
weights_dict = {
    0: weight_const / (129 / 455),
    1: weight_const / (22 / 455),
    2: weight_const / (156 / 455), 
    3: weight_const / (84 / 455),
    4: weight_const / (64 / 455)
}

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn_classification: torch.nn.Module,
               loss_fn_regression: torch.nn.Module,
               loss_fn_ordinal: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               scheduler: torch.optim.lr_scheduler,
               device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    # train_loss, train_acc = 0, 0
    total_class_loss, total_reg_loss, total_ord_loss = 0, 0, 0
    class_acc, reg_acc, ord_acc = 0, 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):

        # Send data to target device
        X, y = X.to(device), y.to(device)

        with torch.autocast(device_type=device, dtype=torch.float16, enabled=True):

            class_out, reg_out, ord_out, enc_out = model(X)

            # 1. One hot encoded for ordinal regression
            num_classes = 5
            y_cumulative = ordinal_labels(y, num_classes)
            y_cumulative = y_cumulative.to(device)


            # with autocast(device_type=device, dtype=torch.float16):
            loss_classification = loss_fn_classification(class_out, y)

            loss_regression = loss_fn_regression(reg_out, y.float())

            loss_ordinal = loss_fn_ordinal(ord_out, y_cumulative)

            total_loss = loss_classification + loss_regression + loss_ordinal
    
        scalar.scale(total_loss).backward()
        scalar.step(optimizer)
        scheduler.step()
        scalar.update()

        # 3. Optimizer zero grad
        optimizer.zero_grad(set_to_none=True)

        # total_loss.backward()
        # optimizer.step()
        # scheduler.step()

        # sample_weights to make up for class imbalance
        # sample_weights = torch.tensor([weights_dict[label] for label in y.cpu().numpy()], dtype=torch.float32).to(device)
        # loss_regression *= sample_weights
        # loss_regression = loss_regression.mean()

        # total_loss = loss_regression

        # train_loss += total_loss.item() 

        # # 3. Optimizer zero grad
        # optimizer.zero_grad(set_to_none=True)

        # 4. Loss backward
        # total_loss.backward()

        # Perform gradient clipping by value
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients

        # 5. Optimizer step
        # optimizer.step()
        # scheduler.step()
        # print(f'last learning rate: {scheduler.get_last_lr()[0]}')

        # Calculate and accumulate accuracy metric across all batches for classification head
        total_class_loss += loss_classification.item()
        y_pred_class = torch.argmax(torch.softmax(class_out, dim=1), dim=1)
        class_acc += (y_pred_class == y).sum().item()/len(y)

        # Calculate and accumulate accuracy metric across all batches for regression head
        total_reg_loss += loss_regression.item()
        y_pred_reg = reg_classify(reg_out, device=device).to(device)
        reg_acc += (y_pred_reg == y).sum().item()/len(y)

        # Calculate and accumulate accuracy metric across all batches for ordinal regression head
        total_ord_loss += loss_ordinal.item()
        # print(f'ord_out shape is: {ord_out.shape}')
        y_pred_ord = torch.sum(torch.round(torch.sigmoid(ord_out)), dim=1, keepdim=True).squeeze(dim=1) - 1
        ord_acc += (y_pred_ord == y).sum().item()/len(y)

        # Calculate and accumulate accuracy metric across all batches
        # y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        # train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    # train_loss = train_loss / len(dataloader)
    # train_acc = train_acc / len(dataloader)
    # print(dataloader)
    total_class_loss = total_class_loss / len(dataloader)
    total_reg_loss = total_reg_loss / len(dataloader)
    total_ord_loss = total_ord_loss / len(dataloader)

    # class_acc /= len(dataloader) 
    # reg_acc /= len(dataloader) 
    # ord_acc /= len(dataloader) 
          
    # print(f'train class acc: {class_acc} | train reg acc: {reg_acc} | train ord acc: {ord_acc}')

    return total_class_loss, total_reg_loss, total_ord_loss

def val_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn_classification: torch.nn.Module,
              loss_fn_regression: torch.nn.Module,
              loss_fn_ordinal: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval() 

    # Setup test loss and test accuracy values
    # test_loss, test_acc = 0, 0
    total_class_loss, total_reg_loss, total_ord_loss = 0, 0, 0
    class_acc, reg_acc, ord_acc = 0, 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            class_out, reg_out, ord_out, enc_out = model(X)
        
            # 1. One hot encoded for ordinal regression
            num_classes = 5
            y_cumulative = ordinal_labels(y, num_classes)
            y_cumulative = y_cumulative.to(device)

            # print(y_cumulative)
            # print(ord_out)

            # 2. Calculate and accumulate loss
            loss_classification = loss_fn_classification(class_out, y)
            loss_regression = loss_fn_regression(reg_out, y.float())
            loss_ordinal = loss_fn_ordinal(ord_out, y_cumulative)

            # loss_ordinal = loss_fn_ordinal(ord_out, y)
            # loss_ordinal = 0.1
        
            # sample_weights to make up for class imbalance
            # sample_weights = torch.tensor([weights_dict[label] for label in y.cpu().numpy()], dtype=torch.float32).to(device)
            # loss_regression *= sample_weights
            # loss_regression = loss_regression.mean()

            total_loss = loss_classification + loss_regression + loss_ordinal
            # total_loss = loss_regression
            # train_loss += total_loss.item() 

            # Calculate and accumulate accuracy metric across all batches for classification head
            total_class_loss += loss_classification.item()
            y_pred_class = torch.argmax(torch.softmax(class_out, dim=1), dim=1)
            class_acc += (y_pred_class == y).sum().item()/len(y)

            # Calculate and accumulate accuracy metric across all batches for regression head
            total_reg_loss += loss_regression.item()
            y_pred_reg = reg_classify(reg_out, device=device).to(device)
            reg_acc += (y_pred_reg == y).sum().item()/len(y)

            # Calculate and accumulate accuracy metric across all batches for ordinal regression head
            total_ord_loss += loss_ordinal.item()
            y_pred_ord = torch.sum(torch.round(torch.sigmoid(ord_out)), dim=1, keepdim=True).squeeze(dim=1) - 1
            ord_acc += (y_pred_ord == y).sum().item()/len(y)
            # Calculate and accumulate accuracy
            # test_pred_labels = test_pred_logits.argmax(dim=1)
            # test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            # print('y_pred_reg is: ', y_pred_reg)
            # print('y is:', y)
            # print('batch reg_acc is: ', reg_acc)

    # Adjust metrics to get average loss and accuracy per batch 

    total_class_loss = total_class_loss / len(dataloader)
    total_reg_loss = total_reg_loss / len(dataloader)
    total_ord_loss = total_ord_loss / len(dataloader)

    class_acc /= len(dataloader) 
    reg_acc /= len(dataloader) 
    ord_acc /= len(dataloader) 

    accs = [class_acc, reg_acc, ord_acc]
    losses = [total_class_loss, total_reg_loss, total_ord_loss]

    return losses, accs

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn_classification: torch.nn.Module,
              loss_fn_regression: torch.nn.Module,
              loss_fn_ordinal: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval() 

    # Setup test loss and test accuracy values
    # test_loss, test_acc = 0, 0
    total_class_loss, total_reg_loss, total_ord_loss = 0, 0, 0
    class_acc, reg_acc, ord_acc = 0, 0, 0
    f1_results = {'f1_class': 0, 'f1_reg': 0, 'f1_ord': 0}


    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            class_out, reg_out, ord_out, enc_out = model(X)
        
            # 1. One hot encoded for ordinal regression
            num_classes = 5
            y_cumulative = ordinal_labels(y, num_classes)
            y_cumulative = y_cumulative.to(device)

            # print('ord_out: ', ord_out)
            # print('y_cumulative: ', y_cumulative)
            # print('y: ', y)

            # print(y_cumulative)
            # print(ord_out)

            # 2. Calculate and accumulate loss
            loss_classification = loss_fn_classification(class_out, y)
            loss_regression = loss_fn_regression(reg_out, y.float())

            # loss ordinal is the mean of binary cross entorpy losses of 5 items. 
            loss_ordinal = loss_fn_ordinal(ord_out, y_cumulative)

            # loss_ordinal = 0.1

            # sample_weights to make up for class imbalance
            # sample_weights = torch.tensor([weights_dict[label] for label in y.cpu().numpy()], dtype=torch.float32).to(device)
            # loss_regression *= sample_weights
            # loss_regression = loss_regression.mean()

            total_loss = loss_classification + loss_regression + loss_ordinal
            # total_loss = loss_regression
            # train_loss += total_loss.item() 

            # Calculate and accumulate accuracy metric across all batches for classification head
            total_class_loss += loss_classification.item()
            y_pred_class = torch.argmax(torch.softmax(class_out, dim=1), dim=1)
            class_acc += (y_pred_class == y).sum().item()/len(y)

            # Calculate and accumulate accuracy metric across all batches for regression head
            total_reg_loss += loss_regression.item()
            y_pred_reg = reg_classify(reg_out, device=device).to(device)
            # print(f'y_pred_reg shape is: {y_pred_reg.shape}')
            # print(f'y shape is: {y.shape}')
            reg_acc += (y_pred_reg == y).sum().item()/len(y)

            # Calculate and accumulate accuracy metric across all batches for ordinal regression head
            total_ord_loss += loss_ordinal.item()
            # print(f'ord_out shape is: {ord_out.shape}')
            y_pred_ord = torch.sum(torch.round(torch.sigmoid(ord_out)), dim=1, keepdim=True).squeeze(dim=1) - 1
            ord_acc += (y_pred_ord == y).sum().item()/len(y)
            # Calculate and accumulate accuracy
            # test_pred_labels = test_pred_logits.argmax(dim=1)
            # test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

            # Calculate F1 score
            # Batch size should be very big so that F1 score is calculated for all test data
            f1_results_batch = calculate_F1_score_multiclass(y_pred_class=y_pred_class.cpu(), y_pred_reg=y_pred_reg.cpu(), y_pred_ord=y_pred_ord.cpu(), y=y.cpu())
            f1_results['f1_class'] += f1_results_batch['f1_class']
            f1_results['f1_reg'] += f1_results_batch['f1_reg']
            f1_results['f1_ord'] += f1_results_batch['f1_ord']


    # Adjust metrics to get average loss and accuracy per batch 
    # test_loss = test_loss / len(dataloader)
    # test_acc = test_acc / len(dataloader)
    # print(dataloader)
    # total_class_loss = total_class_loss / len(dataloader)
    # total_reg_loss = total_reg_loss / len(dataloader)
    # total_ord_loss = total_ord_loss / len(dataloader)

    print(f'f1_class: {f1_results["f1_class"] / len(dataloader)} | f1_reg: {f1_results["f1_reg"] / len(dataloader)} | f1_ord: {f1_results["f1_ord"] / len(dataloader)}')

    class_acc /= len(dataloader) 
    reg_acc /= len(dataloader) 
    ord_acc /= len(dataloader) 
    print(f'test class acc: {class_acc} | test reg acc: {reg_acc} | test ord acc: {ord_acc}')
    return class_acc, reg_out, ord_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler,
          loss_fn_classification: torch.nn.Module,
          loss_fn_regression: torch.nn.Module,
          loss_fn_ordinal: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:

    train_results = {"loss_classification_train": [],
                "loss_regression_train": [],
                "loss_ordinal_train": []}
    val_results = {"loss_classification_val": [],
                "loss_regression_val": [],
                "loss_ordinal_val": []}
    
    # Make sure model on target device
    model.to(device)

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        loss_classification_train, loss_regression_train, loss_ordinal_train = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn_classification=loss_fn_classification,
                                          loss_fn_regression=loss_fn_regression,
                                          loss_fn_ordinal=loss_fn_ordinal,
                                          optimizer=optimizer,
                                          scheduler=scheduler,
                                          device=device)
        
        losses, accs = val_step(model=model,
                                dataloader=val_dataloader,
                                loss_fn_classification=loss_fn_classification,
                                loss_fn_regression=loss_fn_regression,
                                loss_fn_ordinal=loss_fn_ordinal,
                                device=device)
        
        loss_classification_val, loss_regression_val, loss_ordinal_val = losses
        acc_classification_val, acc_regression_val, acc_ordinal_val = accs

        # Print out what's happening
        print(
          f"Epoch: {epoch+1}\n"
          f"loss_classification_train: {loss_classification_train:.4f} | "
          f"loss_regression_train: {loss_regression_train:.4f} | "
          f"loss_ordinal_train: {loss_ordinal_train:.4f}\n"
          f"loss_classification_validation: {loss_classification_val:.4f} | "
          f"loss_regression_validation: {loss_regression_val:.4f} | "
          f"loss_ordinal_validation: {loss_ordinal_val:.4f}\n"
          f"acc_classification_validation: {acc_classification_val:.4f} | "
          f"acc_regression_validation: {acc_regression_val:.4f} | "
          f"acc_ordinal_validation: {acc_ordinal_val:.4f}\n"
        )

        # Update results dictionary
        train_results["loss_classification_train"].append(loss_classification_train)
        train_results["loss_regression_train"].append(loss_regression_train)
        train_results["loss_ordinal_train"].append(loss_ordinal_train)

        val_results["loss_classification_val"].append(loss_classification_val)
        val_results["loss_regression_val"].append(loss_regression_val)
        val_results["loss_ordinal_val"].append(loss_ordinal_val)

        # class_out, reg_out, ord_out, enc_out = model(test_dataloader)
        # print(enc_out)

    # Return the filled results at the end of the epochstrain_
    return train_results, val_results

def pre_train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler,
          loss_fn_classification: torch.nn.Module,
          loss_fn_regression: torch.nn.Module,
          loss_fn_ordinal: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:

    train_results = {"loss_classification_train": [],
                "loss_regression_train": [],
                "loss_ordinal_train": []}
    
    # Make sure model on target device
    model.to(device)

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        loss_classification_train, loss_regression_train, loss_ordinal_train = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn_classification=loss_fn_classification,
                                          loss_fn_regression=loss_fn_regression,
                                          loss_fn_ordinal=loss_fn_ordinal,
                                          optimizer=optimizer,
                                          scheduler=scheduler,
                                          device=device)
        
        # Print out what's happening
        print(
          f"Epoch: {epoch+1}\n"
          f"loss_classification_train: {loss_classification_train:.4f} | "
          f"loss_regression_train: {loss_regression_train:.4f} | "
          f"loss_ordinal_train: {loss_ordinal_train:.4f}\n"
        )

        # Update results dictionary
        train_results["loss_classification_train"].append(loss_classification_train)
        train_results["loss_regression_train"].append(loss_regression_train)
        train_results["loss_ordinal_train"].append(loss_ordinal_train)

        # class_out, reg_out, ord_out, enc_out = model(test_dataloader)
        # print(enc_out)

    # Return the filled results at the end of the epochstrain_
    return train_results

def calculate_F1_score_multiclass(y_pred_class, y_pred_reg, y_pred_ord, y, num_classes=5):

    f1 = MulticlassF1Score(num_classes=num_classes, average='none')  # 'macro', 'micro', or 'weighted', or 'none' for F1 score for each class

    f1_results = {'f1_class': 0, 'f1_reg': 0, 'f1_ord': 0}

    f1_results["f1_class"] = f1(y_pred_class, y)
    f1_results["f1_reg"] = f1(y_pred_reg, y)
    f1_results['f1_ord'] =  f1(y_pred_ord, y)

    # print(f'y_class_pred dim: {y_pred_class.shape}')
    # print(f'y dim: {y.shape}')

    # print(f'F1 score for classification: {f1_class} | F1 score for regression: {f1_reg} | F1 score for ordinal: {f1_ord}')

    return f1_results

def calculate_F1_score_binary(y_pred_class, y_pred_reg, y_pred_ord, y, num_classes=2):

    y[y >= 1] = 1
    y_pred_class[y_pred_class >= 1] = 1
    y_pred_reg[y_pred_reg >= 1] = 1
    y_pred_ord[y_pred_ord >= 1] = 1


    f1 = MulticlassF1Score(num_classes=num_classes, average='macro')  # 'macro', 'micro', or 'weighted', or 'none' for F1 score for each class

    f1_results = {'f1_class': 0, 'f1_reg': 0, 'f1_ord': 0}

    f1_results["f1_class"] = f1(y_pred_class, y)
    f1_results["f1_reg"] = f1(y_pred_reg, y)
    f1_results['f1_ord'] =  f1(y_pred_ord, y)

    return f1_results

