import torch
from torch import nn
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.classification import MulticlassCohenKappa

def reg_classify(x, device):
    bins = torch.tensor([0.5, 1.5, 2.5, 3.5]).to(device)  # Class boundaries
    # Classify using bucketize
    classified = torch.bucketize(x, bins, right=False)  # right=False ensures correct bin placement
    return classified


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               scheduler: torch.optim.lr_scheduler,
               device: torch.device) -> Tuple[float, float]:

    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    loss = 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # print('X is: ', X)
        # print('y float is: ', y.float())

        # out looks like this: tensor([a, b, c, d])
        out = model(X).squeeze(dim=1)
        

        batch_loss = loss_fn(out, y.float())

        # 3. Optimizer zero grad
        optimizer.zero_grad(set_to_none=True)

        # 4. Loss backward
        batch_loss.backward()

        # Perform gradient clipping by value
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients

        # 5. Optimizer step
        optimizer.step()
        scheduler.step()

        # Calculate and accumulate accuracy metric across all batches for classification head
        loss += batch_loss.item()

    # Adjust metrics to get average loss and accuracy per batch 
    loss /= len(dataloader)

    return loss

def val_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    
    # Put model in eval mode
    model.eval() 

    # Setup test loss and test accuracy values
    loss = 0
    acc = 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # out looks like this: tensor([a, b, c, d])
            out = model(X).squeeze(dim=1)
            # print('out is: ', out)

            batch_loss = loss_fn(out, y)

            # Calculate and accumulate accuracy metric across all batches for classification head
            loss += batch_loss.item()

            # Calculate and accumulate accuracy metric across all batches for regression head
            # y_pred looks like this: tensor([a, b, c, d])
            y_pred = reg_classify(out, device=device).to(device)
            # print('y_pred is: ', y_pred)
            # print('y is: ', y)
            acc += (y_pred == y).sum().item()/len(y)
            # print('acc is: ', acc)


    # Adjust metrics to get average loss and accuracy per batch 
    loss /= len(dataloader)
    acc /= len(dataloader) 

    return loss, acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    
    # Put model in eval mode
    model.eval() 

    # Setup test loss and test accuracy values
    acc = 0
    QWK_score = 0
    QWK_metric = MulticlassCohenKappa(num_classes=5)
    temp = 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # out looks like this: tensor([a, b, c, d])
            out = model(X).squeeze(dim=1)

            # Calculate and accumulate accuracy metric across all batches for regression head
            # y_pred looks like this: tensor([a, b, c, d])
            y_pred = reg_classify(out, device=device).to(device)
            acc += (y_pred == y).sum().item()/len(y)

            
            QWK_score += QWK_metric(y_pred, y)

    # Adjust metrics to get average loss and accuracy per batch 
    acc /= len(dataloader) 
    print(f'test acc: {acc}')

    QWK_score = QWK_score.clone()
    QWK_score /= len(dataloader)
    print(f'QWK score: {QWK_score}')

    return acc

def main_model_output(model: torch.nn.Module, 
           dataloader: torch.utils.data.DataLoader,
           device: torch.device):
        
    y_preds = []
    ys = []

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            class_out, reg_out, ord_out, enc_out = model(X)

            y_pred_class = torch.argmax(torch.softmax(class_out, dim=1), dim=1)
            y_pred_reg = reg_classify(reg_out, device=device).to(device)
            y_pred_ord = (torch.sum(torch.round(torch.sigmoid(ord_out)), dim=1, keepdim=True).squeeze(dim=1) - 1)

            for i in range(len(y)):
                y_preds.append([y_pred_class[i].item(), y_pred_reg[i].item(), y_pred_ord[i].item()])
            for i in range(len(y)):
                ys.append(y[i].item())

    y_preds = torch.tensor(y_preds)
    ys = torch.tensor(ys)
    return y_preds, ys

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
    

    train_results = {"loss_train": []}
    val_results = {"loss_val": []}

    # Make sure model on target device
    model.to(device)

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        loss_train = train_step(model=model,
                               dataloader=train_dataloader,
                               loss_fn=loss_fn,
                               optimizer=optimizer,
                               scheduler=scheduler,
                               device=device)
        
        loss_val, acc_val = val_step(model=model,
                                  dataloader=val_dataloader,
                                  loss_fn=loss_fn,
                                  device=device)
        
        # Print out what's happening
        print(
          f"Epoch: {epoch}\n"
          f"loss_train: {loss_train:.4f} | "
          f"loss_val: {loss_val:.4f} | "
          f"acc_val: {acc_val:.4f}\n"
        )

        # Update results dictionary
        train_results["loss_train"].append(loss_train)

        val_results["loss_val"].append(loss_val)


    # Return the filled results at the end of the epochstrain_
    return train_results, val_results