"""
A series of helper functions used throughout the course.

If a function gets defined once and could be used over and over, it'll go in here.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from matplotlib.patches import Patch

from torch import nn

import os
import zipfile

from pathlib import Path

import requests

# Walk through an image classification directory and find out how many files (images)
# are in each subdirectory.
import os

def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    Args:
    dir_path (str): target directory

    Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Plots decision boundaries of model predicting on X in comparison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


# Plot linear data or training and test and predictions (optional)
def plot_predictions(
    train_data, train_labels, test_data, test_labels, predictions=None
):
    """
  Plots linear training data and test data and compares predictions.
  """
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})


# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def print_train_time(start, end, device=None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time


# Plot loss curves of a model

def plot_loss_curves_post_train(train_results, val_results):

    epochs = range(len(train_results["loss_train"]))

    plt.figure(figsize=(18, 9))
    plt.plot(epochs, train_results['loss_train'], label="loss_train", color='blue')
    plt.plot(epochs, val_results['loss_val'], label="loss_val", color='blue', linestyle='dotted')
    plt.xlabel("Epochs")
    plt.legend(loc='upper right')
    plt.show()

def plot_loss_curves_main_train(train_results, val_results):

    epochs = range(len(train_results["loss_classification_train"]))

    plt.figure(figsize=(18, 9))
    plt.plot(epochs, train_results['loss_classification_train'], label="loss_classification_train", color='blue')
    plt.plot(epochs, val_results['loss_classification_val'], label="loss_classification_val", color='blue', linestyle='dotted')

    plt.plot(epochs, train_results['loss_regression_train'], label="loss_regression_train", color='red')
    plt.plot(epochs, val_results['loss_regression_val'], label="loss_regression_val", color='red', linestyle='dotted')

    plt.plot(epochs, train_results['loss_ordinal_train'], label="loss_ordinal_train", color='green')
    plt.plot(epochs, val_results['loss_ordinal_val'], label="loss_ordinal_val", color='green', linestyle='dotted')
    plt.xlabel("Epochs")
    plt.legend(loc='upper right')
    plt.show()

def plot_loss_curves_pre_train(train_results):

    epochs = range(len(train_results["loss_classification_train"]))

    plt.figure(figsize=(18, 9))
    plt.plot(epochs, train_results['loss_classification_train'], label="loss_classification_train", color='blue')

    plt.plot(epochs, train_results['loss_regression_train'], label="loss_regression_train", color='red')

    plt.plot(epochs, train_results['loss_ordinal_train'], label="loss_ordinal_train", color='green')
    plt.xlabel("Epochs")
    plt.legend(loc='upper right')
    plt.show()

# Pred and plot image function from notebook 04
# See creation: https://www.learnpytorch.io/04_pytorch_custom_datasets/#113-putting-custom-image-prediction-together-building-a-function
from typing import List
import torchvision


"""
Utility functions to make predictions.

Main reference for code creation: https://www.learnpytorch.io/06_pytorch_transfer_learning/#6-make-predictions-on-images-from-the-test-set 
"""
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

from typing import List, Tuple

from PIL import Image

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Predict on a target image with a target model
# Function created in: https://www.learnpytorch.io/06_pytorch_transfer_learning/#6-make-predictions-on-images-from-the-test-set
def pred_and_plot_image(
    model: torch.nn.Module,
    class_names: List[str],
    image_path: str,
    image_name: str,
    image_size: Tuple[int, int] = (224, 224),
    transform: torchvision.transforms = None,
    device: torch.device = device,
):
    """Predicts on a target image with a target model.

    Args:
        model (torch.nn.Module): A trained (or untrained) PyTorch model to predict on an image.
        class_names (List[str]): A list of target classes to map predictions to.
        image_path (str): Filepath to target image to predict on.
        image_size (Tuple[int, int], optional): Size to transform target image to. Defaults to (224, 224).
        transform (torchvision.transforms, optional): Transform to perform on image. Defaults to None which uses ImageNet normalization.
        device (torch.device, optional): Target device to perform prediction on. Defaults to device.
    """

    # Open image
    img = Image.open(image_path)

    # Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    ### Predict on image ###

    # Make sure the model is on the target device
    model.to(device)

    # Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
        transformed_image = image_transform(img).unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(transformed_image.to(device))

    # Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    df = pd.read_csv('../idrid/idrid_labels.csv')
    first_column_name = df.columns[0]
    matched_rows = df[df[first_column_name] == image_name]
    actual_label = matched_rows.iloc[0]['diagnosis']
    
    # Plot image with predicted label and probability
    plt.figure()
    plt.imshow(img)
    plt.title(
        f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f} | Actual: {class_names[actual_label]}"
    )
    plt.axis(False)

def set_seeds(seed: int=42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)

def plot_t_SNE(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        NUM_ITER: int = 2000
        ):
    model.eval()  # Set to evaluation mode
    features, labels = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            class_out, reg_out, ord_out, enc_out = model(X)  # Extract last-layer features
            features.append(enc_out.cpu().numpy())  # Move to CPU
            labels.append(y.numpy())

    features = np.concatenate(features, axis=0)  # Convert list to array
    labels = np.concatenate(labels, axis=0)

    # Apply t-SNE
    perp_vals = [10,20]
    for perp in perp_vals:
        tsne = TSNE(n_components=2, perplexity=perp, random_state=45)
        features_2d = tsne.fit_transform(features)

        # Class labels
        class_labels = {
            0: 'No DR',
            1: 'Mild DR',
            2: 'Moderate DR',
            3: 'Severe DR',
            4: 'Proliferative DR'
        }

        cmap = plt.cm.jet
        norm = plt.Normalize(0, 4)

        # Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap=cmap, norm=norm)

        legend_elements = [
            Patch(facecolor=cmap(norm(i)), edgecolor='black', label=class_labels[i]) for i in range(5)
        ]

        # Add legend outside top-right
        plt.legend(
            handles=legend_elements,
            title="DR Stage",
            loc='upper left',
            bbox_to_anchor=(1.01, 1),
            labelspacing=1,      
            borderaxespad=0.5,    
        )

        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.title("t-SNE Visualization of Model's Features (Extracted by Encoder)")

        plt.tight_layout()
        plt.show()