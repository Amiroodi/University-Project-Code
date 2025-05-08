import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
import albumentations as A
import cv2
from sklearn.model_selection import KFold
from torch.utils.data import ConcatDataset


IDRID_image_folder = "../../IDRID/Imagenes/Imagenes" 
IDRID_csv_file = "../../IDRID/idrid_labels.csv"  

MESSIDOR_image_folder = "../../MESSIDOR/images"
MESSIDOR_csv_file = "../../MESSIDOR/messidor_data.csv"

APTOS_train_image_folder = "../../APTOS/resized_train_19"
APTOS_train_csv_file = "../../APTOS/labels/trainLabels19.csv"  

APTOS_test_image_folder = "../../APTOS/resized_test_15"
APTOS_test_csv_file = "../../APTOS/labels/testLabels15.csv"  

NUM_WORKERS = 4

class LoadDataset_IDRID(Dataset):
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
    
class LoadDataset_MESSIDOR(Dataset):
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
        img_path = os.path.join(self.image_folder, img_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        if self.transform:
            image = self.transform(image=image)["image"]
            
        return image, label
    
class LoadDataset_APOTS(Dataset):
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
    
def create_train_val_dataloader(
    transform: A.Compose,
    batch_size: int, 
    shrink_size=None,
    num_workers: int=NUM_WORKERS,
    ):
  
    # combine all datasets
    train_dataset_1 = LoadDataset_IDRID(IDRID_image_folder, IDRID_csv_file, transform=transform)
    train_dataset_2 = LoadDataset_MESSIDOR(MESSIDOR_image_folder, MESSIDOR_csv_file, transform=transform)
    train_dataset_3 = LoadDataset_APOTS(APTOS_train_image_folder, APTOS_train_csv_file, transform=transform)
    combined_dataset = ConcatDataset([train_dataset_1, train_dataset_2, train_dataset_3])

    # Shrinking dataset size for test purposes
    if shrink_size is not None:
        combined_dataset = Subset(combined_dataset, range(shrink_size))

    train_val_dataloader = []

    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # train_idx and val_idx are indexes of selected items in train_dataset
    for fold, (train_idx, val_idx) in enumerate(kf.split(combined_dataset)):
        train_subset = Subset(combined_dataset, train_idx)
        val_subset = Subset(combined_dataset, val_idx)

        train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True)
        val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True)

        fold = {
                'train_dataloader': train_dataloader,
                'val_dataloader': val_dataloader
               }
        train_val_dataloader.append(fold)

    # Get class names
    class_names = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']

    return train_val_dataloader, class_names

def create_test_dataloader(
    transform: A.Compose,
    batch_size: int, 
    shrink_size=None,
    num_workers: int=NUM_WORKERS,
    ):
  
    # Use ImageFolder to create dataset(s)
    test_dataset = LoadDataset_APOTS(APTOS_test_image_folder, APTOS_test_csv_file, transform=transform)

    # Shrinking dataset size for test purposes
    if shrink_size is not None:
        test_dataset = Subset(test_dataset, range(shrink_size))

    # Get class names
    class_names = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    return test_dataloader, class_names

def create_train_dataloader(
    transform: A.Compose,
    batch_size: int, 
    shrink_size=None,
    num_workers: int=NUM_WORKERS,
    ):
  
    # combine all datasets
    train_dataset_1 = LoadDataset_IDRID(IDRID_image_folder, IDRID_csv_file, transform=transform)
    train_dataset_2 = LoadDataset_MESSIDOR(MESSIDOR_image_folder, MESSIDOR_csv_file, transform=transform)
    train_dataset_3 = LoadDataset_APOTS(APTOS_train_image_folder, APTOS_train_csv_file, transform=transform)
    combined_dataset = ConcatDataset([train_dataset_1, train_dataset_2, train_dataset_3])

    # Shrinking dataset size for test purposes
    if shrink_size is not None:
        combined_dataset = Subset(combined_dataset, range(shrink_size))

    # Get class names
    class_names = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']

    train_dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    return train_dataloader, class_names