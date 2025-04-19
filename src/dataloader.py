import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from utils import transforming
from collections import Counter


class PeopleDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.labels_df = pd.read_csv(os.path.join(data_dir, 'train_answers.csv'))
        self.img_dir = os.path.join(data_dir, 'img_train')
        self.image_files = [f for f in os.listdir(self.img_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        img_id = os.path.splitext(img_name)[0]
        label = self.labels_df[self.labels_df['img_id'].astype(str) == img_id]['target_feature'].values[0]
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label


def get_class_weights(dataset):
    targets = [dataset[i][1].item() for i in range(len(dataset))]
    class_counts = Counter(targets)
    total_samples = len(targets)
    num_classes = len(class_counts)

    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in targets]

    return torch.DoubleTensor(sample_weights)


def get_transforms(augmentation_type=None):
    print(f"Augmentation type: {augmentation_type}")
    if augmentation_type == "basic":
        return transforming.basic_augmentation
    elif augmentation_type == "advanced":
        return transforming.advanced_augmentation
    else:
        return transforming.basic_transformation


def setup_data_loaders(batch_size, train_set, valid_set=None, num_workers=4, use_sampler=False):
    if use_sampler:
        sample_weights = get_class_weights(train_set)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers
        )
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True, 
            num_workers=num_workers
        )

    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    ) if valid_set is not None else None

    return train_loader, valid_loader


def print_batch_shape(data_loader: DataLoader, loader_type: str, ):
    valid_batch = next(iter(data_loader))
    images_valid, labels_valid = valid_batch
    print(f"{loader_type} batch shape: {images_valid.shape}")


def split_dataset(dataset, valid_ratio=0.25):
    total_size = len(dataset)
    valid_size = int(total_size * valid_ratio)
    train_size = total_size - valid_size
    return random_split(dataset, [train_size, valid_size])


if __name__ == "__main__":

    data_dir = "PATH TO YOUR DATA"
    transforms = get_transforms()

    full_dataset = PeopleDataset(data_dir, transform=transforms)

    train_set, valid_set = split_dataset(full_dataset, valid_ratio=0.2)

    batch_size = 32
    train_loader, valid_loader = setup_data_loaders(
        batch_size=batch_size,
        train_set=train_set,
        valid_set=valid_set
    )

    print_batch_shape(train_loader, "Train")
    if valid_loader:
        print_batch_shape(valid_loader, "Validation")
