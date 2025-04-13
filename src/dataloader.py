import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


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


def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((288, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((288, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def setup_data_loaders(batch_size, train_set, valid_set=None, num_workers=4):
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
    train_transforms = get_train_transforms()
    val_transforms = get_val_transforms()

    data_dir = "PATH TO YOUR DATA"
    full_dataset = PeopleDataset(data_dir)

    train_set, valid_set = split_dataset(full_dataset, valid_ratio=0.2)

    train_set.dataset.transform = train_transforms
    valid_set.dataset.transform = val_transforms

    batch_size = 32
    train_loader, valid_loader = setup_data_loaders(
        batch_size=batch_size,
        train_set=train_set,
        valid_set=valid_set
    )

    print_batch_shape(train_loader, "Train")
    if valid_loader:
        print_batch_shape(valid_loader, "Validation")
