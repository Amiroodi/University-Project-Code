"""
Contains functions for training and testing a PyTorch model.
"""

import torch
from torch import nn
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from torchmetrics.classification import MulticlassF1Score

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

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn_classification: torch.nn.Module,
               loss_fn_regression: torch.nn.Module,
               loss_fn_ordinal: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               scheduler: torch.optim.lr_scheduler,
               device: torch.device) -> Tuple[float, float]:

    # Put model in train mode
    model.train()

    total = 0
    total_class_loss, total_reg_loss, total_ord_loss = 0, 0, 0
    correct_class, correct_reg, correct_ord = 0, 0, 0

    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # One hot encoded for ordinal regression
        num_classes = 5
        y_cumulative = ordinal_labels(y, num_classes)
        y_cumulative = y_cumulative.to(device)

        with torch.autocast(device_type=device, dtype=torch.float16, enabled=True):

            class_out, reg_out, ord_out, enc_out, final_out = model(X)
            # print('reg out:', reg_out)

            loss_classification = loss_fn_classification(class_out, y)
            loss_regression = loss_fn_regression(reg_out, y.float())
           
            # loss ordinal is the mean of binary cross entorpy losses of 5 items. 
            loss_ordinal = loss_fn_ordinal(ord_out, y_cumulative)

            total_loss = loss_classification + loss_regression + loss_ordinal
    
        scalar.scale(total_loss).backward()
        scalar.step(optimizer)
        scheduler.step()
        scalar.update()

        # 3. Optimizer zero grad
        optimizer.zero_grad(set_to_none=True)

        total += y.size(0)

        # Calculate and accumulate accuracy metric across all batches for classification head
        total_class_loss += loss_classification.item()
        y_pred_class = torch.argmax(torch.softmax(class_out, dim=1), dim=1)
        correct_class += (y_pred_class == y).sum().item()
        # class_acc += (y_pred_class == y).sum().item()/len(y)

        # Calculate and accumulate accuracy metric across all batches for regression head
        total_reg_loss += loss_regression.item()
        y_pred_reg = reg_classify(reg_out, device=device).to(device)
        correct_reg += (y_pred_reg == y).sum().item()
        # reg_acc += (y_pred_reg == y).sum().item()/len(y)

        # Calculate and accumulate accuracy metric across all batches for ordinal regression head
        total_ord_loss += loss_ordinal.item()
        y_pred_ord = torch.sum(torch.round(torch.sigmoid(ord_out)), dim=1, keepdim=True).squeeze(dim=1) - 1
        correct_ord += (y_pred_ord == y).sum().item()
        # ord_acc += (y_pred_ord == y).sum().item()/len(y)

    # Adjust metrics to get average loss and accuracy per batch 
    total_class_loss = total_class_loss / len(dataloader)
    total_reg_loss = total_reg_loss / len(dataloader)
    total_ord_loss = total_ord_loss / len(dataloader)

    class_acc = correct_class / total
    reg_acc = correct_reg / total
    ord_acc = correct_ord / total

    accs = [class_acc, reg_acc, ord_acc]
    losses = [total_class_loss, total_reg_loss, total_ord_loss]

    return losses, accs

def val_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn_classification: torch.nn.Module,
              loss_fn_regression: torch.nn.Module,
              loss_fn_ordinal: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:

    # Put model in eval mode
    model.eval() 

    total_class_loss, total_reg_loss, total_ord_loss = 0, 0, 0
    class_acc, reg_acc, ord_acc = 0, 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            class_out, reg_out, ord_out, enc_out, fianl_out = model(X)
        
            # 1. One hot encoded for ordinal regression
            num_classes = 5
            y_cumulative = ordinal_labels(y, num_classes)
            y_cumulative = y_cumulative.to(device)

            # 2. Calculate and accumulate loss
            loss_classification = loss_fn_classification(class_out, y)
            loss_regression = loss_fn_regression(reg_out, y.float())
                
            # loss ordinal is the mean of binary cross entorpy losses of 5 items. 
            loss_ordinal = loss_fn_ordinal(ord_out, y_cumulative)

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

    # Put model in eval mode
    model.eval() 

    total_class_loss, total_reg_loss, total_ord_loss = 0, 0, 0
    class_acc, reg_acc, ord_acc = 0, 0, 0
    f1_results = {'f1_class': 0, 'f1_reg': 0, 'f1_ord': 0}

    # Turn on inference context manager
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            class_out, reg_out, ord_out, enc_out, final_out = model(X)
            # print('reg out:', reg_out)

            # 1. One hot encoded for ordinal regression
            num_classes = 5
            y_cumulative = ordinal_labels(y, num_classes)
            y_cumulative = y_cumulative.to(device)

            # 2. Calculate and accumulate loss
            loss_classification = loss_fn_classification(class_out, y)
            loss_regression = loss_fn_regression(reg_out, y.float())

            # loss ordinal is the mean of binary cross entorpy losses of 5 items. 
            loss_ordinal = loss_fn_ordinal(ord_out, y_cumulative)

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

            # Calculate F1 score
            # Batch size should be very big so that F1 score is calculated for all test data
            f1_results_batch = calculate_F1_score_multiclass(y_pred_class=y_pred_class.cpu(), y_pred_reg=y_pred_reg.cpu(), y_pred_ord=y_pred_ord.cpu(), y=y.cpu())
            f1_results['f1_class'] += f1_results_batch['f1_class']
            f1_results['f1_reg'] += f1_results_batch['f1_reg']
            f1_results['f1_ord'] += f1_results_batch['f1_ord']

    print(f'f1_class: {f1_results["f1_class"] / len(dataloader)} | f1_reg: {f1_results["f1_reg"] / len(dataloader)} | f1_ord: {f1_results["f1_ord"] / len(dataloader)}')
    
    # Adjust metrics to get average loss and accuracy per batch 
    class_acc /= len(dataloader) 
    reg_acc /= len(dataloader) 
    ord_acc /= len(dataloader) 

    print(f'test class acc: {class_acc} | test reg acc: {reg_acc} | test ord acc: {ord_acc}')
    return class_acc, reg_out, ord_acc

def train(
    model: torch.nn.Module, 
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader, 
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    loss_fn_classification: torch.nn.Module,
    loss_fn_regression: torch.nn.Module,
    loss_fn_ordinal: torch.nn.Module,
    epochs: int,
    device: torch.device) -> Dict[str, List]:

    train_results = {
        "loss_classification_train": [],
        "loss_regression_train": [],
        "loss_ordinal_train": []}
    val_results = {
        "loss_classification_val": [],
        "loss_regression_val": [],
        "loss_ordinal_val": []}
    
    # Make sure model on target device
    model.to(device)

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_losses, train_accs = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn_classification=loss_fn_classification,
            loss_fn_regression=loss_fn_regression,
            loss_fn_ordinal=loss_fn_ordinal,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device)

        val_losses, val_accs = val_step(
            model=model,
            dataloader=val_dataloader,
            loss_fn_classification=loss_fn_classification,
            loss_fn_regression=loss_fn_regression,
            loss_fn_ordinal=loss_fn_ordinal,
            device=device)
        
        loss_classification_train, loss_regression_train, loss_ordinal_train = train_losses
        acc_classification_train, acc_regression_train, acc_ordinal_train = train_accs

        loss_classification_val, loss_regression_val, loss_ordinal_val = val_losses
        acc_classification_val, acc_regression_val, acc_ordinal_val = val_accs

        # Print out what's happening
        if epoch % 5 == 0:
            print(
                f"Epoch: {epoch}\n"
                f"loss_classification_train: {loss_classification_train:.4f} | "
                f"loss_regression_train: {loss_regression_train:.4f} | "
                f"loss_ordinal_train: {loss_ordinal_train:.4f}\n"
                f"loss_classification_validation: {loss_classification_val:.4f} | "
                f"loss_regression_validation: {loss_regression_val:.4f} | "
                f"loss_ordinal_validation: {loss_ordinal_val:.4f}\n"
                f"acc_classification_validation: {acc_classification_val:.4f} | "
                f"acc_regression_validation: {acc_regression_val:.4f} | "
                f"acc_ordinal_validation: {acc_ordinal_val:.4f}\n")

        # Update results dictionary
        train_results["loss_classification_train"].append(loss_classification_train)
        train_results["loss_regression_train"].append(loss_regression_train)
        train_results["loss_ordinal_train"].append(loss_ordinal_train)

        val_results["loss_classification_val"].append(loss_classification_val)
        val_results["loss_regression_val"].append(loss_regression_val)
        val_results["loss_ordinal_val"].append(loss_ordinal_val)

    return train_results, val_results

def pre_train(
    model: torch.nn.Module, 
    train_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    loss_fn_classification: torch.nn.Module,
    loss_fn_regression: torch.nn.Module,
    loss_fn_ordinal: torch.nn.Module,
    epochs: int,
    device: torch.device) -> Dict[str, List]:

    train_results = {
        "loss_classification_train": [],
        "loss_regression_train": [],
        "loss_ordinal_train": []}
    
    # Make sure model on target device
    model.to(device)

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        loss_classification_train, loss_regression_train, loss_ordinal_train = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn_classification=loss_fn_classification,
            loss_fn_regression=loss_fn_regression,
            loss_fn_ordinal=loss_fn_ordinal,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device)
        
        # Print out what's happening
        if epoch % 4 == 0:
            print(
                f"Epoch: {epoch}\n"
                f"loss_classification_train: {loss_classification_train:.4f} | "
                f"loss_regression_train: {loss_regression_train:.4f} | "
                f"loss_ordinal_train: {loss_ordinal_train:.4f}\n")

        # Update results dictionary
        train_results["loss_classification_train"].append(loss_classification_train)
        train_results["loss_regression_train"].append(loss_regression_train)
        train_results["loss_ordinal_train"].append(loss_ordinal_train)

    return train_results

def calculate_F1_score_multiclass(y_pred_class, y_pred_reg, y_pred_ord, y, num_classes=5):

    f1 = MulticlassF1Score(num_classes=num_classes, average='none')  # 'macro', 'micro', or 'weighted', or 'none' for F1 score for each class

    f1_results = {'f1_class': 0, 'f1_reg': 0, 'f1_ord': 0}

    # print(y_pred_reg)
    # print(y)

    f1_results["f1_class"] = f1(y_pred_class, y)
    f1_results["f1_reg"] = f1(y_pred_reg, y)
    f1_results['f1_ord'] =  f1(y_pred_ord, y)

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

