import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
import albumentations as A
import cv2

APTOS_train_image_folder = "../APTOS/resized_train_15"
APTOS_train_csv_file = "../APTOS/labels/trainLabels15.csv"  

APTOS_test_image_folder = "../APTOS/resized_test_15"
APTOS_test_csv_file = "../APTOS/labels/testLabels15.csv"  

NUM_WORKERS = 4
    
class LoadDataset(Dataset):
    def __init__(self, image_folder, csv_file, transform=None):
        self.image_folder = image_folder
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get image filename and label from the DataFrame
        img_name = self.df.iloc[idx, 0]  # Assuming first column is filename
        label = self.df.iloc[idx, 1]  # Assuming second column is label (0-4)

        # Load image
        img_path = os.path.join(self.image_folder, img_name) + '.jpg'
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        if self.transform:
            image = self.transform(image=image)["image"]
        
        return image, label
    
def create_train_dataloader(
    transform: A.Compose,
    batch_size: int, 
    shrink_size=None,
    num_workers: int=NUM_WORKERS,
    ):
  
    train_dataset = LoadDataset(APTOS_train_image_folder, APTOS_train_csv_file, transform=transform)

    # Shrinking dataset size for test purposes
    if shrink_size is not None:
        train_dataset = Subset(train_dataset, range(shrink_size))

    # Get class names
    class_names = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True)

    return train_dataloader, class_names

def create_test_dataloader(
    transform: A.Compose,
    batch_size: int, 
    shrink_size=None,
    num_workers: int=NUM_WORKERS,
    ):
  
    test_dataset = LoadDataset(APTOS_test_image_folder, APTOS_test_csv_file, transform=transform)

    # Shrinking dataset size for test purposes
    if shrink_size is not None:
        test_dataset = Subset(test_dataset, range(shrink_size))

    # Get class names
    class_names = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True)

    return test_dataloader, class_names