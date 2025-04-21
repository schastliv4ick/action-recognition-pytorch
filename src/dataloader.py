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
        self.image_files = []
        self.labels = []
        self.class_names = ['sports', 'inactivity quiet/light', 'miscellaneous', 'occupation', 'water activities',
                            'home activities', 'lawn and garden', 'religious activities', 'winter activities',
                            'conditioning exercise', 'bicycling', 'fishing and hunting', 'dancing', 'walking',
                            'running',
                            'self care', 'home repair', 'volunteer activities', 'music playing', 'transportation']
        self._load_data()
        self.initial_class_counts = Counter(self.labels)
        self.class_to_index = {cls_name: i for i, cls_name in enumerate(self.class_names)}  # Corrected mapping
        self.index_to_class = {i: cls_name for i, cls_name in enumerate(self.class_names)}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        label_str = self.labels[idx]  # Get the string label
        label_int = self.class_to_index[label_str]  # Convert to integer using the mapping
        label = torch.tensor(label_int, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label
    def get_num_classes(self):
        """Returns the number of unique classes in the dataset after filtering."""
        return len(set(self.labels))

    def _load_data(self):
        """Loads image file paths and their corresponding string labels."""
        for img_name in os.listdir(self.img_dir):
            if img_name.endswith('.jpg'):
                img_id = os.path.splitext(img_name)[0]
                label_index = self.labels_df[self.labels_df['img_id'].astype(str) == img_id]['target_feature'].values[0]
                label_str = self.class_names[label_index]  # Get the string label
                self.image_files.append(img_name)
                self.labels.append(label_str)  # Store string label

    def print_class_distribution(self):
        """Prints the number of samples and proportion for each class in the dataset."""
        class_counts = Counter(self.labels)
        total_samples = len(self.labels)
        print("Class Distribution:")
        for i, class_name in enumerate(self.class_names):
            count = class_counts.get(i, 0)  # Get count, default to 0 if class not present
            proportion = count / total_samples if total_samples > 0 else 0
            print(f"  Class '{class_name}': {count} samples ({proportion:.4f})")

    def filter_by_min_threshold(self, min_threshold):
        """Filters the dataset to keep only classes with a proportion of samples
        greater than or equal to the min_threshold.
        """
        total_samples = len(self.labels)
        filtered_image_files = []
        filtered_labels_strings = []
        excluded_classes_log = {}
        original_numerical_labels = list(self.labels)  # Keep the numerical labels for iteration

        class_counts = Counter()
        for label_index in original_numerical_labels:
            class_name = self.class_names[label_index]
            class_counts[class_name] += 1

        class_proportions = {}
        for class_name in self.class_names:
            count = class_counts.get(class_name, 0)
            class_proportion = count / total_samples if total_samples > 0 else 0
            class_proportions[class_name] = class_proportion

        indices_to_keep = []
        for idx, label_index in enumerate(original_numerical_labels):
            class_name = self.class_names[label_index]  # Transform index to name
            if class_proportions[class_name] >= min_threshold:
                indices_to_keep.append(idx)

        self.image_files = [self.image_files[i] for i in indices_to_keep]
        # Now, update self.labels to contain the *numerical* labels of the kept samples
        self.labels = [self.labels[i] for i in indices_to_keep]

        # Identify and log excluded classes
        for i, class_name in enumerate(self.class_names):
            class_proportion = class_proportions.get(class_name, 0)
            original_count = class_counts.get(class_name, 0)
            if class_proportion < min_threshold and original_count > 0:
                excluded_classes_log[class_name] = original_count

        print(f"Filtering by minimum threshold ({min_threshold:.4f}):")
        if excluded_classes_log:
            for cls, count in excluded_classes_log.items():
                proportion = class_proportions[cls]
                print(f"  Excluded images of class '{cls}' with {count} samples (proportion: {proportion:.4f}).")
        else:
            print("  No classes excluded based on the minimum proportion threshold.")
        print(f"Dataset size reduced to {len(self.labels)} from {total_samples} samples initially.")
        print(f"  Number of images after filtering: {len(self.image_files)}")
        print(f"  Number of labels after filtering: {len(self.labels)}")
        print(f"  Unique labels after filtering: {len(set(self.labels))}")

    def filter_by_classes(self, classes_to_exclude):
        """Filters the dataset to remove samples of specified classes and logs excluded classes."""
        original_length = len(self.labels)
        filtered_image_files = []
        filtered_labels = []
        excluded_classes_log = {}

        for img_file, label_str in zip(self.image_files, self.labels):
            if label_str not in classes_to_exclude:
                filtered_image_files.append(img_file)
                filtered_labels.append(label_str)
            elif label_str in classes_to_exclude and label_str not in excluded_classes_log:
                excluded_classes_log[label_str] = self.labels.count(label_str)

        self.image_files = filtered_image_files
        self.labels = filtered_labels

        # Rebuild class_names and class_to_index
        self.class_names = sorted(list(set(self.labels)))
        self.class_to_index = {cls_name: i for i, cls_name in enumerate(self.class_names)}
        self.index_to_class = {i: cls_name for i, cls_name in enumerate(self.class_names)}
        print(f"Number of classes after filtering: {len(self.class_names)}")

        print(f"Filtering by explicitly excluded classes ({classes_to_exclude}):")
        if excluded_classes_log:
            for cls, count in excluded_classes_log.items():
                print(f"  Excluded images of class '{cls}' with {count} samples.")
        else:
            print("  No specified classes found in the dataset to exclude.")
        print(f"Dataset size reduced to {len(self.labels)} from {original_length} samples initially.")

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
