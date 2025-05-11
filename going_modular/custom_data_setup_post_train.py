import torch
from torch.utils.data import Dataset, DataLoader, Subset
from going_modular import engine_post_train, custom_data_setup_main_train
from going_modular.ThreeHeadCNN import ThreeHeadCNN
import albumentations as A
from sklearn.model_selection import KFold

NUM_WORKERS = 4

class LoadDataset(Dataset):
    def __init__(self, model, dataloader, device):
        y_preds, ys = engine_post_train.main_model_output(model=model, dataloader=dataloader, device=device)        
        self.y_preds = y_preds
        self.ys = ys

    def __len__(self):
        return len(self.ys)
    
    def __getitem__(self, idx):
        y_pred = self.y_preds[idx]
        y = self.ys[idx]
        return y_pred, y
    
def create_train_val_dataloader(
    transform: A.Compose,
    batch_size: int, 
    device,
    shrink_size: int,
    num_workers: int=NUM_WORKERS,
    ):

    model = ThreeHeadCNN().to(device)
    # load trained model's weights
    model.load_state_dict(torch.load("models/main_train_model.pth",weights_only=True, map_location=device))

    train_dataloader, class_names = custom_data_setup_main_train.create_train_dataloader(
        batch_size=32, # batch size is not importants here
        ) 
    
    train_dataset = LoadDataset(model=model, dataloader=train_dataloader, device=device)

    # Shrinking dataset size for test purposes
    if shrink_size is not None:
        train_dataset = Subset(train_dataset, range(shrink_size))

    train_val_dataloader = []

    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # train_idx and val_idx are indexes of selected items in train_dataset
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)

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
    device,
    shrink_size: int,
    num_workers: int=NUM_WORKERS,
    ):

    model = ThreeHeadCNN().to(device)
    # load trained model's weights
    model.load_state_dict(torch.load("models/main_train_model.pth", weights_only=True, map_location=device))

    test_dataloader, class_names = custom_data_setup_main_train.create_test_dataloader(
        batch_size=32, # batch size is not importants here
        )
    
    test_dataset = LoadDataset(model=model, dataloader=test_dataloader, device=device)

    # Shrinking dataset size for test purposes
    if shrink_size is not None:
        test_dataset = Subset(test_dataset, range(shrink_size))

    # Get class names
    class_names = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True)

    return test_dataloader, class_names

def create_train_dataloader(
    transform: A.Compose,
    batch_size: int, 
    device,
    shrink_size: int,
    num_workers: int=NUM_WORKERS,
    ):

    model = ThreeHeadCNN().to(device)
    # load trained model's weights
    model.load_state_dict(torch.load("models/main_train_model.pth", weights_only=True, map_location=device))

    train_dataloader, class_names = custom_data_setup_main_train.create_train_dataloader(
        batch_size=32, # batch size is not importants here
        )
    
    train_dataset = LoadDataset(model=model, dataloader=train_dataloader, device=device)

    # Shrinking dataset size for test purposes
    if shrink_size is not None:
        train_dataset = Subset(train_dataset, range(shrink_size))

    # Get class names
    class_names = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True)

    return train_dataloader, class_names