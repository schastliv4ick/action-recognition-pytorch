import matplotlib.pyplot as plt
#import torch # For checking tensor type
import dataloader
from dataloader import PeopleDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    # Геометрические преобразования
    A.Rotate(limit=30, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomScale(scale_limit=0.2, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
    
    # Цветовые преобразования
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.2, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    
    # Пространственные искажения
    A.Perspective(scale=(0.05, 0.1), p=0.3),
    A.RandomCrop(width=224, height=224, p=0.5),
    
    # Имитация реальных условий
    A.MotionBlur(blur_limit=7, p=0.3),
    A.Downscale(scale_min=0.7, scale_max=0.9, p=0.2),
    
    # Преобразование в тензор
    ToTensorV2()
])

def visualize_augmentations(dataset, idx=0, samples=3):
    # Make a copy of the transform list to modify for visualization
    if isinstance(dataset.transform, A.Compose):
        vis_transform_list = [
            t for t in dataset.transform
            if not isinstance(t, (A.Normalize, A.ToTensorV2))
        ]
        vis_transform = A.Compose(vis_transform_list)
    else:
        # Handle cases where transform might not be Compose (optional)
        print("Warning: Could not automatically strip Normalize/ToTensor for visualization.")
        vis_transform = dataset.transform
 
    figure, ax = plt.subplots(1, samples + 1, figsize=(12, 5))
 
    # --- Get the original image --- #
    # Temporarily disable transform to get raw image
    original_transform = dataset.transform
    dataset.transform = None
    image, label = dataset[idx]
    dataset.transform = original_transform # Restore original transform
 
    # Display original
    ax[0].imshow(image)
    ax[0].set_title("Original")
    ax[0].axis("off")
 
    # --- Apply and display augmented versions --- #
    for i in range(samples):
        # Apply the visualization transform
        if vis_transform:
            augmented = vis_transform(image=image)
            aug_image = augmented['image']
        else:
             # Should not happen if dataset had a transform
            aug_image = image
 
        ax[i+1].imshow(aug_image)
        ax[i+1].set_title(f"Augmented {i+1}")
        ax[i+1].axis("off")
 
    plt.tight_layout()
    plt.show()

PATH_TO_DATA = "D:\\VS\\ml\\human_poses_data\\yandex-ml-2025\\data"
full_dataset = PeopleDataset(PATH_TO_DATA)
# Assuming train_dataset is created with train_transform:
visualize_augmentations(full_dataset, samples=10)
