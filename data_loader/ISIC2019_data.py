import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import skimage.io as io
from skimage import color

class ISIC2019Dataset(Dataset):
    """ISIC2019 dataset loader.
    
    This dataset contains skin lesion images with demographic information.
    The dataset is used for skin lesion classification with fairness considerations.
    
    Args:
        df: DataFrame containing image metadata
        root_dir: Directory containing the images
        transform: Optional transform to be applied on images
    """
    def __init__(self, df: pd.DataFrame, root_dir: str, transform: transforms.Compose = None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Load image
        img_name = os.path.join(self.root_dir, f"{self.df.iloc[idx]['image']}.jpg")
        image = io.imread(img_name)
        if len(image.shape) < 3:
            image = color.gray2rgb(image)
            
        # Get labels
        sex = self.df.iloc[idx]['sex_id']
        age = self.df.iloc[idx]['age_approx']
        
        # Convert to binary attributes
        sex_binary = 1 if sex == 1 else 0
        age_binary = 1 if age > 55 else 0
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        return {
            'image': image,
            'class_id': self.df.iloc[idx]['class_id'],
            'sex_id': sex,
            'sex_color_binary': sex_binary,
            'age_approx': age,
            'age_approx_binary': age_binary
        }

def get_weighted_sampler(df: pd.DataFrame, label_column: str) -> WeightedRandomSampler:
    """Create a weighted sampler for balanced training.
    
    Args:
        df: DataFrame containing the data
        label_column: Column name for the labels
        
    Returns:
        WeightedRandomSampler for balanced sampling
    """
    class_counts = df[label_column].value_counts()
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in df[label_column]]
    return WeightedRandomSampler(
        weights=torch.FloatTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

def get_transforms(is_training: bool, image_size: int = 128, crop_size: int = 112) -> transforms.Compose:
    """Get image transforms for training or testing.
    
    Args:
        is_training: Whether to use training transforms
        image_size: Size to resize images to
        crop_size: Size to crop images to
        
    Returns:
        Compose transform
    """
    if is_training:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
        ])

def ISIC2019_dataloader_score(
    batch_size: int,
    workers: int,
    image_dir: str = 'ISIC2019/dataset_images',
    csv_file: str = 'ISIC2019/ISIC2019.csv',
    root: str = '',
    fair_attr: str = 'gender',
    ctype: str = 'class_id'
) -> tuple:
    """Create dataloaders for ISIC2019 dataset.
    
    Args:
        batch_size: Batch size for dataloaders
        workers: Number of workers for dataloaders
        image_dir: Directory containing images
        csv_file: Path to CSV file with metadata
        root: Root directory for data
        fair_attr: Fairness attribute to use ('gender' or 'age')
        ctype: Column name for classification labels
        
    Returns:
        Tuple of (train_loader, male_loader, female_loader, val_loader, test_loader)
    """
    # Load and split data
    df = pd.read_csv(csv_file)
    train_df, val_df, test_df = np.split(
        df.sample(frac=1, random_state=42),
        [int(0.8*len(df)), int(0.9*len(df))]
    )
    
    # Split training data by fairness attribute
    if fair_attr == 'gender':
        male_df = train_df[train_df['sex_id'] == 1]
        female_df = train_df[train_df['sex_id'] == 0]
    else:  # age
        male_df = train_df[train_df['age_approx'] > 55]
        female_df = train_df[train_df['age_approx'] <= 55]
    
    # Create transforms
    train_transform = get_transforms(is_training=True)
    test_transform = get_transforms(is_training=False)
    
    # Create datasets
    train_dataset = ISIC2019Dataset(train_df, image_dir, train_transform)
    male_dataset = ISIC2019Dataset(male_df, image_dir, train_transform)
    female_dataset = ISIC2019Dataset(female_df, image_dir, train_transform)
    val_dataset = ISIC2019Dataset(val_df, image_dir, test_transform)
    test_dataset = ISIC2019Dataset(test_df, image_dir, test_transform)
    
    # Create dataloaders
    kwargs = {'num_workers': workers, 'pin_memory': True} if torch.cuda.is_available() else {}
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=get_weighted_sampler(train_df, ctype),
        **kwargs
    )
    
    male_loader = DataLoader(
        male_dataset,
        batch_size=batch_size,
        sampler=get_weighted_sampler(male_df, ctype),
        **kwargs
    )
    
    female_loader = DataLoader(
        female_dataset,
        batch_size=batch_size,
        sampler=get_weighted_sampler(female_df, ctype),
        **kwargs
    )
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    
    return train_loader, male_loader, female_loader, val_loader, test_loader


