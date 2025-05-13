"""
Contains functions for training and testing a PyTorch model.
"""

import torch
from torch import nn
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.classification import MulticlassCohenKappa
from torchmetrics import CohenKappa

scalar = torch.amp.GradScaler('cuda', enabled=True)

def reg_classify(x, device):
    bins = torch.tensor([0.5, 1.5, 2.5, 3.5]).to(device)  # Class boundaries
    # Classify using bucketize
    classified = torch.bucketize(x, bins, right=False)  # right=False ensures correct bin placement
    return classified

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:

    # Put model in train mode
    model.train()

    total_loss = 0
    acc = 0

    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        with torch.autocast(device_type=device, dtype=torch.float16, enabled=True):

            class_out, reg_out, ord_out, enc_out, final_out = model(X)
            # print('final out:', final_out)


            loss = loss_fn(final_out, y.float())
            # print('losss is: ', loss)
            # print('final_out is :', final_out)
            # print('float is : ', final_out.float())

        scalar.scale(loss).backward()
        scalar.step(optimizer)
        scalar.update()

        # 3. Optimizer zero grad
        optimizer.zero_grad(set_to_none=True)

        # Calculate and accumulate accuracy metric across all batches for final head
        total_loss += loss.item()
        y_pred_final = reg_classify(final_out, device=device).to(device)
        acc += (y_pred_final == y).sum().item()/len(y)

    # Adjust metrics to get average loss and accuracy per batch 
    total_loss = total_loss / len(dataloader)

    return total_loss

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:

    # Put model in eval mode
    model.eval() 

    total_loss = 0
    acc = 0
    total_f1_score_per_class, total_f1_score_macro = 0, 0

    QWK_score = 0
    QWK_metric = MulticlassCohenKappa(num_classes=5, weights='quadratic').to(device)
    # QWK_metric = CohenKappa(num_classes=5, weights='quadratic')

    # Turn on inference context manager
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            class_out, reg_out, ord_out, enc_out, final_out = model(X)
            # print('final out:', final_out)

        
            loss = loss_fn(final_out, y)

            # Calculate and accumulate accuracy metric across all batches for final head
            total_loss += loss.item()
            y_pred_final = reg_classify(final_out, device=device).to(device)
            acc += (y_pred_final == y).sum().item()/len(y)

            # Calculate F1 score
            # Batch size should be very big so that F1 score is calculated for all test data
            # print(final_out)
            # print(y_pred_final.squeeze(dim=1))
            # print(y)
            f1_score_per_class, f1_score_macro = calculate_F1_score_multiclass(y_pred_final=y_pred_final.cpu(), y=y.cpu())

            total_f1_score_per_class += f1_score_per_class
            total_f1_score_macro += f1_score_macro

            QWK_score += QWK_metric(y_pred_final, y)

    print(f'f1_score_per_class: {total_f1_score_per_class / len(dataloader)}')
    print(f'f1_score_macro (unweighted average): {total_f1_score_macro / len(dataloader)}')

    # Adjust metrics to get average loss and accuracy per batch 
    acc /= len(dataloader) 
    print(f'test acc: {acc}')

    QWK_score = QWK_score.clone()
    QWK_score /= len(dataloader)
    print(f'QWK score: {QWK_score}')

    return acc

def train(
    model: torch.nn.Module, 
    train_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device) -> Dict[str, List]:

    train_results = {
        "loss_train": []
    }
    
    # Make sure model on target device
    model.to(device)

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        loss_train= train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device)
        
        # Print out what's happening
        print(
            f"Epoch: {epoch+1}\n"
            f"loss_train: {loss_train:.4f}\n")

        # Update results dictionary
        train_results["loss_train"].append(loss_train)

    return train_results

def calculate_F1_score_multiclass(y_pred_final, y, num_classes=5):

    f1_per_class = MulticlassF1Score(num_classes=num_classes, average='none')  # 'macro', 'micro', or 'weighted', or 'none' for F1 score for each class
    f1_macro = MulticlassF1Score(num_classes=num_classes, average='macro')  # 'macro', 'micro', or 'weighted', or 'none' for F1 score for each class


    f1_result_per_class = f1_per_class(y_pred_final, y)
    f1_result_macro = f1_macro(y_pred_final, y)

    return f1_result_per_class, f1_result_macro

def calculate_F1_score_binary(y_pred_final, y, num_classes=2):

    y[y >= 1] = 1
    y_pred_final[y_pred_final >= 1] = 1

    f1 = MulticlassF1Score(num_classes=num_classes, average='macro')  # 'macro', 'micro', or 'weighted', or 'none' for F1 score for each class

    f1_result = f1(y_pred_final, y)

    return f1_result

