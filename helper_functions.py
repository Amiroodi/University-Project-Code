"""
A series of helper functions used throughout the course.
"""

import torch
from torch import nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from matplotlib.patches import Patch
import albumentations as A
from albumentations.pytorch import ToTensorV2


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

def get_augmentation_A_transforms(p):
    A_transforms = A.Compose([ 
        A.Resize(240, 240),
        A.OpticalDistortion(distort_limit=0.3, p=p),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=p),
        A.ElasticTransform(alpha=40, sigma=50, p=p),
        A.Affine(scale=[0.7, 1.4], translate_percent=[-0.05, 0.05], shear=[-15, 15], rotate=[-45, 45], p=p),
        A.HorizontalFlip(p=p), 
        A.VerticalFlip(p=p), 
        A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=p),  
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=p),  
        A.AdditiveNoise(noise_type='gaussian', spatial_mode='shared', approximation=1.0, noise_params={"mean_range": (0.0, 0.0), "std_range": (0.02, 0.05)}, p=p),
        A.GaussianBlur(blur_limit=1, p=p), 
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=p),  
        A.Emboss(alpha=(0.5, 0.6), strength=(0.6, 0.7), p=p),  
        A.RandomGamma(gamma_limit=(80, 120), p=p),  
        A.CoarseDropout(num_holes_range=(1, 2), hole_height_range=(0.1, 0.2), hole_width_range=(0.1, 0.2), fill=0, fill_mask=None, p=p),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.ToFloat(),
        ToTensorV2()], seed=33)
    
    return A_transforms

def get_augmentation_no_transforms():
    no_transforms = A.Compose([
    A.Resize(240, 240),       
    A.ToFloat(),
    ToTensorV2()], seed=33)

    return no_transforms
