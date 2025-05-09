import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
import albumentations as A
import cv2
from sklearn.model_selection import KFold
from torch.utils.data import ConcatDataset
from sklearn.model_selection import StratifiedShuffleSplit


IDRID_image_folder = "../../IDRID/Imagenes/Imagenes" 
IDRID_csv_file = "../../IDRID/idrid_labels.csv"  

MESSIDOR_image_folder = "../../MESSIDOR/images"
MESSIDOR_csv_file = "../../MESSIDOR/messidor_data.csv"

APTOS_train_image_folder = "../../APTOS/resized_train_19"
APTOS_train_csv_file = "../../APTOS/labels/trainLabels19.csv"  

APTOS_test_image_folder = "../../APTOS/resized_test_15"
APTOS_test_csv_file = "../../APTOS/labels/testLabels15.csv"  

NUM_WORKERS = 0

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
        if self.image_folder == MESSIDOR_image_folder:
            img_path = os.path.join(self.image_folder, img_name)
        else:
            img_path = os.path.join(self.image_folder, img_name) + '.jpg'

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        if self.transform:
            image = self.transform(image=image)["image"]
            
        return image, label
    
def LoadDataset_train_test_split(transform, shrink_size):
    train_dataset_1 = LoadDataset(IDRID_image_folder, IDRID_csv_file, transform=transform)
    train_dataset_2 = LoadDataset(MESSIDOR_image_folder, MESSIDOR_csv_file, transform=transform)
    train_dataset_3 = LoadDataset(APTOS_train_image_folder, APTOS_train_csv_file, transform=transform)
    combined_dataset = ConcatDataset([train_dataset_1, train_dataset_2, train_dataset_3])

    # Shrinking dataset size for test purposes
    if shrink_size is not None:
        combined_dataset = Subset(combined_dataset, range(shrink_size))

    labels = [combined_dataset[i][1] for i in range(len(combined_dataset))]  # assuming (image, label)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(range(len(combined_dataset)), labels))

    train_dataset = Subset(combined_dataset, train_idx)
    test_dataset = Subset(combined_dataset, test_idx)

    print(len(train_dataset))
    print(len(test_dataset))

    print(train_dataset)
    print(test_dataset)

    return train_dataset, test_dataset
        
def create_train_val_dataloader(
    transform: A.Compose,
    batch_size: int, 
    shrink_size=None,
    num_workers: int=NUM_WORKERS,
    ):
  
    train_dataset, _ = LoadDataset_train_test_split(transform=transform, shrink_size=shrink_size)

    train_val_dataloader = []

    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # train_idx and val_idx are indexes of selected items in train_dataset
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)

        train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=False, pin_memory=True)
        val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=False, pin_memory=True)

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
  
    _, test_dataset = LoadDataset_train_test_split(transform=transform, shrink_size=shrink_size)

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
  
    train_dataset, _ = LoadDataset_train_test_split(transform=transform, shrink_size=shrink_size)

    # Get class names
    class_names = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=False, pin_memory=True)

    return train_dataloader, class_names