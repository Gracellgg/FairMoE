import os, random
from time import sleep
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import skimage
from skimage import io, color
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from .data_utils import AutoAugment, Cutout_v0

def read_fitzpatrick17k_dataset_metainfo(csv_file_name='fitzpatrick17k_no_histo/fitzpatrick17k_no_histo.csv', dev_mode=None):
    """Read and process Fitzpatrick17k dataset metadata.
    
    Args:
        csv_file_name: Path to the CSV file
        dev_mode: If "dev", use a smaller sample for development
        
    Returns:
        Processed DataFrame with labels and metadata
    """
    if dev_mode == "dev":
        df = pd.read_csv(csv_file_name).sample(1000)
    else:
        df = pd.read_csv(csv_file_name)
    
    # Convert labels to category codes
    df["low"] = df['label'].astype('category').cat.codes
    df["mid"] = df['nine_partition_label'].astype('category').cat.codes
    df["high"] = df['three_partition_label'].astype('category').cat.codes
    df["hasher"] = df["md5hash"]
    df["image_path"] = df["url"]
    return df

def fitzpatrick17k_holdout(df, holdout_sets, ctype):
    """
    df: dataframe of the whole set
    holdout_sets: how to split the data for training and testing
    """
    total_classes = len(df.label.unique())
    for holdout_set in holdout_sets:
        print("holdout policy: {:}".format(holdout_set))
        if holdout_set == "expert_select":
            df2 = df
            train = df2[df2.qc.isnull()]
            test = df2[df2.qc=="1 Diagnostic"]
        elif holdout_set == "random_holdout":
            train, test, y_train, y_test = train_test_split(
                                                df,
                                                eval("df." + ctype),
                                                test_size=0.2,
                                                random_state=4242,
                                                stratify=eval("df." + ctype))
            print(train['fitzpatrick'].value_counts().sort_index())
            print(test['fitzpatrick'].value_counts().sort_index())
        elif holdout_set == "dermaamin":
            combo = set(df[df.image_path.str.contains("dermaamin")==True].label.unique()) & set(df[df.image_path.str.contains("dermaamin")==False].label.unique())
            df2 = df[df.label.isin(combo)]
            df2["low"] = df2['label'].astype('category').cat.codes
            train = df2[df2.image_path.str.contains("dermaamin") == False]
            test = df2[df2.image_path.str.contains("dermaamin") == True]
            print("# common labels: {:}".format(len(combo)))
        elif holdout_set == "br":
            combo = set(df[df.image_path.str.contains("dermaamin")==True].label.unique()) & set(df[df.image_path.str.contains("dermaamin")==False].label.unique())
            df2 = df[df.label.isin(combo)]
            df2["low"] = df2['label'].astype('category').cat.codes
            train = df2[df2.image_path.str.contains("dermaamin") == True]
            test = df2[df2.image_path.str.contains("dermaamin") == False]
            print("# common labels: {:}".format(len(combo)))
            print(train.label.nunique())
            print(test.label.nunique())
        elif holdout_set == "a12":
            train = df[(df.fitzpatrick==1)|(df.fitzpatrick==2)]
            test = df[(df.fitzpatrick!=1)&(df.fitzpatrick!=2)]
            combo = set(train.label.unique()) & set(test.label.unique())
            print("# common labels: {:}/{:}({:}+{:})".format(len(combo), total_classes, len(train.label.unique()), len(test.label.unique())))
            # print(combo)
            train = train[train.label.isin(combo)].reset_index()
            test = test[test.label.isin(combo)].reset_index()
            train["low"] = train['label'].astype('category').cat.codes
            test["low"] = test['label'].astype('category').cat.codes
        elif holdout_set == "a34":
            train = df[(df.fitzpatrick==3)|(df.fitzpatrick==4)]
            test = df[(df.fitzpatrick!=3)&(df.fitzpatrick!=4)]
            combo = set(train.label.unique()) & set(test.label.unique())
            print("# common labels: {:}/{:}({:}+{:})".format(len(combo), total_classes, len(train.label.unique()), len(test.label.unique())))
            train = train[train.label.isin(combo)].reset_index()
            test = test[test.label.isin(combo)].reset_index()
            train["low"] = train['label'].astype('category').cat.codes
            test["low"] = test['label'].astype('category').cat.codes
        elif holdout_set == "a56":
            train = df[(df.fitzpatrick==5)|(df.fitzpatrick==6)]
            test = df[(df.fitzpatrick!=5)&(df.fitzpatrick!=6)]
            combo = set(train.label.unique()) & set(test.label.unique())
            print("# common labels: {:}/{:}({:}+{:})".format(len(combo), total_classes, len(train.label.unique()), len(test.label.unique())))
            train = train[train.label.isin(combo)].reset_index()
            test = test[test.label.isin(combo)].reset_index()
            train["low"] = train['label'].astype('category').cat.codes
            test["low"] = test['label'].astype('category').cat.codes
        elif holdout_set == "a123":
            train = df[(df.fitzpatrick==4)|(df.fitzpatrick==5)|(df.fitzpatrick==6)]
            test = df[(df.fitzpatrick!=4)&(df.fitzpatrick!=5)&(df.fitzpatrick!=6)]
            combo = set(train.label.unique()) & set(test.label.unique())
            print("# common labels: {:}/{:}({:}+{:})".format(len(combo), total_classes, len(train.label.unique()), len(test.label.unique())))
            train = train[train.label.isin(combo)].reset_index()
            test = test[test.label.isin(combo)].reset_index()
            train["low"] = train['label'].astype('category').cat.codes
            test["low"] = test['label'].astype('category').cat.codes
        elif holdout_set == "a456":
            train = df[(df.fitzpatrick==1)|(df.fitzpatrick==2)|(df.fitzpatrick==3)]
            test = df[(df.fitzpatrick!=1)&(df.fitzpatrick!=2)&(df.fitzpatrick!=3)]
            combo = set(train.label.unique()) & set(test.label.unique())
            print("# common labels: {:}/{:}({:}+{:})".format(len(combo), total_classes, len(train.label.unique()), len(test.label.unique())))
            train = train[train.label.isin(combo)].reset_index()
            test = test[test.label.isin(combo)].reset_index()
            train["low"] = train['label'].astype('category').cat.codes
            test["low"] = test['label'].astype('category').cat.codes

    return train, test

def fitzpatrick17k_holdout_score(df, holdout_sets, ctype, root=''):
    for holdout_set in holdout_sets:
        print("holdout policy: {:}".format(holdout_set))
        try:
            train_and_score = read_fitzpatrick17k_dataset_metainfo(csv_file_name=root + "train_score.csv") # 8931 light + 3862 dark 8857 + 3919; target: 800 + 800
            val = read_fitzpatrick17k_dataset_metainfo(csv_file_name=root + "all_val.csv")
            test = read_fitzpatrick17k_dataset_metainfo(csv_file_name=root + "all_test.csv")
            print('find existing files')
        except:
            train_and_score, val_and_test, _, _ = train_test_split(
                                            df,
                                            eval("df." + ctype),
                                            test_size=0.2,
                                            random_state=4242,
                                            stratify=eval("df." + ctype))
            train, score, _, _ = train_test_split(
                                            train_and_score,
                                            eval("train_and_score." + ctype),
                                            test_size=0.125,
                                            random_state=4242,
                                            stratify=eval("train_and_score." + ctype))
            val, test, _, _ = train_test_split(
                                            val_and_test,
                                            eval("val_and_test." + ctype),
                                            test_size=0.5,
                                            random_state=4242,
                                            stratify=eval("val_and_test." + ctype))
            train.to_csv(root + "all_train.csv")
            score.to_csv(root + "all_score.csv")
            val.to_csv(root + "all_val.csv")
            test.to_csv(root + "all_test.csv")
            train_and_score.to_csv(root + "train_score.csv")
            train = read_fitzpatrick17k_dataset_metainfo(csv_file_name=root + "all_train.csv")
            score = read_fitzpatrick17k_dataset_metainfo(csv_file_name=root + "all_score.csv")

        try:
            light_data = read_fitzpatrick17k_dataset_metainfo(csv_file_name=root + "light_score.csv")
            dark_data = read_fitzpatrick17k_dataset_metainfo(csv_file_name=root + "dark_score.csv")
        except:
            light_data = train_and_score
            dark_data = train_and_score
            light_data = light_data.drop(light_data[(light_data["fitzpatrick"] < 1) | (light_data["fitzpatrick"] > 3)].index)
            #light_data = light_data.sample(n=800)
            light_data.to_csv(root + "light_score.csv")
            dark_data = dark_data.drop(dark_data[(dark_data["fitzpatrick"] < 4) | (dark_data["fitzpatrick"] > 6)].index)
            #dark_data = dark_data.sample(n=800)
            dark_data.to_csv(root + "dark_score.csv")
            score = pd.concat([light_data, dark_data])
            score.to_csv(root + "all_score.csv")

        try:
            skin_type1 = read_fitzpatrick17k_dataset_metainfo(csv_file_name=root + "skin_type1.csv")
            skin_type2 = read_fitzpatrick17k_dataset_metainfo(csv_file_name=root + "skin_type2.csv")
            skin_type3 = read_fitzpatrick17k_dataset_metainfo(csv_file_name=root + "skin_type3.csv")
            skin_type4 = read_fitzpatrick17k_dataset_metainfo(csv_file_name=root + "skin_type4.csv")
            skin_type5 = read_fitzpatrick17k_dataset_metainfo(csv_file_name=root + "skin_type5.csv")
            skin_type6 = read_fitzpatrick17k_dataset_metainfo(csv_file_name=root + "skin_type6.csv")
            skin_type_list = [skin_type1, skin_type2, skin_type3, skin_type4, skin_type5, skin_type6]
        except:
            skin_type1, skin_type2, skin_type3, skin_type4, skin_type5, skin_type6 = train_and_score, train_and_score, train_and_score, train_and_score, train_and_score, train_and_score
            skin_type1 = skin_type1[skin_type1["fitzpatrick"] == 1]
            skin_type2 = skin_type2[skin_type2["fitzpatrick"] == 2]
            skin_type3 = skin_type3[skin_type3["fitzpatrick"] == 3]
            skin_type4 = skin_type4[skin_type4["fitzpatrick"] == 4]
            skin_type5 = skin_type5[skin_type5["fitzpatrick"] == 5]
            skin_type6 = skin_type6[skin_type6["fitzpatrick"] == 6]
            skin_type1.to_csv(root + "skin_type1.csv")
            skin_type2.to_csv(root + "skin_type2.csv")
            skin_type3.to_csv(root + "skin_type3.csv")
            skin_type4.to_csv(root + "skin_type4.csv")
            skin_type5.to_csv(root + "skin_type5.csv")
            skin_type6.to_csv(root + "skin_type6.csv")
            print("number of skin type 1: {:}".format(len(skin_type1)))
            print("number of skin type 2: {:}".format(len(skin_type2)))
            print("number of skin type 3: {:}".format(len(skin_type3)))
            print("number of skin type 4: {:}".format(len(skin_type4)))
            print("number of skin type 5: {:}".format(len(skin_type5)))
            print("number of skin type 6: {:}".format(len(skin_type6)))

            skin_type_list = [skin_type1, skin_type2, skin_type3, skin_type4, skin_type5, skin_type6]
        # train[ctype].value_counts().sort_index() # 70%
        # score[ctype].value_counts().sort_index() # ~10%
        train_and_score[ctype].value_counts().sort_index() # 80%
        val[ctype].value_counts().sort_index() # 10%
        test[ctype].value_counts().sort_index() # 10%
        light_data[ctype].value_counts().sort_index() # ~5% (800)
        dark_data[ctype].value_counts().sort_index() # ~5% (800)

    return train_and_score, light_data, dark_data, skin_type_list, val, test

class ISIC2019_Augmentations():
    def __init__(self, is_training, image_size=256, input_size=224):
        mdlParams = dict()
        
        mdlParams['setMean'] = np.array([0.0, 0.0, 0.0])   
        mdlParams['setStd'] = np.array([1.0, 1.0, 1.0])
        self.image_size = image_size
        mdlParams['input_size'] = [input_size, input_size, 3]
        self.input_size = (np.int32(mdlParams['input_size'][0]),np.int32(mdlParams['input_size'][1]))   

        # training augmentations
        if is_training:
            self.same_sized_crop = True
            self.only_downsmaple = False

            # For Fitzpatrik17k dataset
            #mdlParams['full_color_distort'] = True
            mdlParams['autoaugment'] = True     
            mdlParams['flip_lr_ud'] = True
            mdlParams['full_rot'] = 180
            mdlParams['scale'] = (0.8,1.2)
            mdlParams['shear'] = 10
            mdlParams['cutout'] = 16

            # For PAD-UFES-20 dataset
            # mdlParams['autoaugment'] = False     
            # mdlParams['flip_lr_ud'] = True
            # mdlParams['full_rot'] = 180
            # mdlParams['scale'] = (0.9,1.1)
            # mdlParams['shear'] = 10
            # mdlParams['cutout'] = 10
            
            # For celebA dataset
            # mdlParams['autoaugment'] = False     
            # mdlParams['flip_lr_ud'] = True
            # mdlParams['full_rot'] = 180
            # mdlParams['scale'] = (0.95,1.05)
            # mdlParams['shear'] = 5
            # mdlParams['cutout'] = 10

            transforms = self.get_train_augmentations(mdlParams)
        else:
            # test augmentations
            transforms = self.get_test_augmentations(mdlParams)
        self.transforms = transforms
    
    def get_test_augmentations(self, mdlParams):
        all_transforms = [
                transforms.ToPILImage(),
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(np.float32(mdlParams['setMean']),np.float32(mdlParams['setStd']))]
        composed = transforms.Compose(all_transforms)
        return composed

    def get_train_augmentations(self, mdlParams):
        all_transforms = [transforms.ToPILImage()]
        # Normal train proc
        if self.same_sized_crop:
            all_transforms.append(transforms.Resize(self.image_size))
            all_transforms.append(transforms.RandomCrop(self.input_size))
        elif self.only_downsmaple:
            all_transforms.append(transforms.Resize(self.input_size))
        else:
            all_transforms.append(transforms.RandomResizedCrop(self.input_size[0],scale=(mdlParams.get('scale_min',0.08),1.0)))
        if mdlParams.get('flip_lr_ud',False):
            all_transforms.append(transforms.RandomHorizontalFlip())
            all_transforms.append(transforms.RandomVerticalFlip())
        # Full rot
        if mdlParams.get('full_rot',0) > 0:
            if mdlParams.get('scale',False):
                all_transforms.append(transforms.RandomChoice([transforms.RandomAffine(mdlParams['full_rot'], scale=mdlParams['scale'], shear=mdlParams.get('shear',0), resample=Image.NEAREST),
                                                            transforms.RandomAffine(mdlParams['full_rot'],scale=mdlParams['scale'],shear=mdlParams.get('shear',0), resample=Image.BICUBIC),
                                                            transforms.RandomAffine(mdlParams['full_rot'],scale=mdlParams['scale'],shear=mdlParams.get('shear',0), resample=Image.BILINEAR)])) 
            else:
                all_transforms.append(transforms.RandomChoice([transforms.RandomRotation(mdlParams['full_rot'], resample=Image.NEAREST),
                                                            transforms.RandomRotation(mdlParams['full_rot'], resample=Image.BICUBIC),
                                                            transforms.RandomRotation(mdlParams['full_rot'], resample=Image.BILINEAR)]))    
        # Color distortion
        if mdlParams.get('full_color_distort') is not None:
            all_transforms.append(transforms.ColorJitter(brightness=mdlParams.get('brightness_aug',32. / 255.),saturation=mdlParams.get('saturation_aug',0.5), contrast = mdlParams.get('contrast_aug',0.5), hue = mdlParams.get('hue_aug',0.2)))
        else:
            all_transforms.append(transforms.ColorJitter(brightness=32. / 255.,saturation=0.5))   
        # Autoaugment
        if mdlParams.get('autoaugment',False):
            all_transforms.append(AutoAugment())             
        # Cutout
        if mdlParams.get('cutout',0) > 0:
            all_transforms.append(Cutout_v0(n_holes=1,length=mdlParams['cutout']))                             
        # Normalize
        all_transforms.append(transforms.ToTensor())
        all_transforms.append(transforms.Normalize(np.float32(mdlParams['setMean']),np.float32(mdlParams['setStd'])))            
        # All transforms
        composed = transforms.Compose(all_transforms)         

        return composed

class Fitzpatrick17kDataset(Dataset):
    """Fitzpatrick17k dataset loader.
    
    This dataset contains skin lesion images with Fitzpatrick skin type annotations.
    The dataset is used for skin lesion classification with fairness considerations.
    
    Args:
        df: DataFrame containing image metadata
        root_dir: Directory containing the images
        transform: Optional transform to be applied on images
        ctype: Column name for classification labels
    """
    def __init__(self, df: pd.DataFrame, root_dir: str, transform: transforms.Compose = None, ctype: str = 'low'):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.ctype = ctype

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Load image
        img_name = os.path.join(self.root_dir, f"{self.df.iloc[idx]['hasher']}.jpg")
        try:
            image = io.imread(img_name)
            if len(image.shape) < 3:
                image = color.gray2rgb(image)
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            raise
            
        # Get labels
        fitzpatrick = self.df.iloc[idx]['fitzpatrick']
        skin_color_binary = 0 if 1 <= fitzpatrick <= 3 else 1
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        return {
            'image': image,
            'skin_color_binary': skin_color_binary,
            'fitzpatrick': fitzpatrick,
            'class_id': self.df.iloc[idx][self.ctype],
            'low': self.df.iloc[idx]['low'],
            'mid': self.df.iloc[idx]['mid'],
            'high': self.df.iloc[idx]['high']
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

def fitzpatrick17k_dataloader_score(
    batch_size: int,
    workers: int,
    image_dir: str = 'fitzpatrick17k/dataset_images',
    csv_file: str = 'fitzpatrick17k/fitzpatrick17k.csv',
    root: str = '',
    ctype: str = 'class_id'
) -> tuple:
    """Create dataloaders for Fitzpatrick17k dataset.
    
    Args:
        batch_size: Batch size for dataloaders
        workers: Number of workers for dataloaders
        image_dir: Directory containing images
        csv_file: Path to CSV file with metadata
        root: Root directory for data
        ctype: Column name for classification labels
        
    Returns:
        Tuple of (train_loader, light_loader, dark_loader, skin_type_loaders, val_loader, test_loader)
    """
    # Load and process data
    df = read_fitzpatrick17k_dataset_metainfo(csv_file_name=csv_file)
    
    # Split data
    train_df, val_df, test_df = np.split(
        df.sample(frac=1, random_state=42),
        [int(0.8*len(df)), int(0.9*len(df))]
    )
    
    # Split training data into light and dark skin types
    light_df = train_df[train_df['fitzpatrick'].between(1, 3)]
    dark_df = train_df[train_df['fitzpatrick'].between(4, 6)]
    
    # Create skin type specific datasets
    skin_type_dfs = [
        train_df[train_df['fitzpatrick'] == i]
        for i in range(1, 7)
    ]
    
    # Create transforms
    train_transform = get_transforms(is_training=True)
    test_transform = get_transforms(is_training=False)
    
    # Create datasets
    train_dataset = Fitzpatrick17kDataset(train_df, image_dir, train_transform, ctype)
    light_dataset = Fitzpatrick17kDataset(light_df, image_dir, train_transform, ctype)
    dark_dataset = Fitzpatrick17kDataset(dark_df, image_dir, train_transform, ctype)
    skin_type_datasets = [
        Fitzpatrick17kDataset(df, image_dir, train_transform, ctype)
        for df in skin_type_dfs
    ]
    val_dataset = Fitzpatrick17kDataset(val_df, image_dir, test_transform, ctype)
    test_dataset = Fitzpatrick17kDataset(test_df, image_dir, test_transform, ctype)
    
    # Create dataloaders
    kwargs = {'num_workers': workers, 'pin_memory': True} if torch.cuda.is_available() else {}
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=get_weighted_sampler(train_df, ctype),
        **kwargs
    )
    
    light_loader = DataLoader(
        light_dataset,
        batch_size=batch_size,
        sampler=get_weighted_sampler(light_df, ctype),
        **kwargs
    )
    
    dark_loader = DataLoader(
        dark_dataset,
        batch_size=batch_size,
        sampler=get_weighted_sampler(dark_df, ctype),
        **kwargs
    )
    
    skin_type_loaders = [
        DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=get_weighted_sampler(df, ctype),
            **kwargs
        )
        for dataset, df in zip(skin_type_datasets, skin_type_dfs)
    ]
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    
    return train_loader, light_loader, dark_loader, skin_type_loaders, val_loader, test_loader

# Add alias for backward compatibility
fitzpatric17k_dataloader_score = fitzpatrick17k_dataloader_score

if __name__ == "__main__":
    df = read_fitzpatrick17k_dataset_metainfo(csv_file_name='fitzpatrick17k/fitzpatrick17k.csv')
    for index, row in df.iterrows():
        try:
            os.system("wget " + row["image_path"] + " -O " + os.path.join("fitzpatrick17k", "dataset_images", row["hasher"] + ".jpg"))
        except:
            print("FAIL:", row["hasher"])
