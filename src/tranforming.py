import matplotlib.pyplot as plt
import torch  # For checking tensor type
import dataloader
from dataloader import PeopleDataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import random

class RandomAdjustColor:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, img):
        if random.random() < 0.5:
            img = F.adjust_brightness(img, random.uniform(1 - self.brightness, 1 + self.brightness))
        if random.random() < 0.5:
            img = F.adjust_contrast(img, random.uniform(1 - self.contrast, 1 + self.contrast))
        if random.random() < 0.5:
            img = F.adjust_saturation(img, random.uniform(1 - self.saturation, 1 + self.saturation))
        if random.random() < 0.5:
            img = F.adjust_hue(img, random.uniform(-self.hue, self.hue))
        return img
    
# Определение аугментации данных
transform = transforms.Compose([
    # Случайное горизонтальное отражение
    transforms.RandomHorizontalFlip(p=0.5),
    
    # Случайное вращение изображения (до 30 градусов)
    transforms.RandomRotation(degrees=30),
    
    # Случайное изменение размера и обрезка
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
    
    # Случайное изменение цветовых характеристик
    RandomAdjustColor(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    
    # Преобразование в тензор PyTorch
    transforms.ToTensor(),
    
    # Нормализация изображения (стандартные значения для ImageNet)
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Путь к данным
PATH_TO_DATA = "D:\\VS\\ml\\human_poses_data\\yandex-ml-2025\\data"



# Создание датасета с аугментацией
aug_dataset = PeopleDataset(PATH_TO_DATA, transform=transform)

# Визуализация аугментированных изображений
plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    img, lab = aug_dataset[i]  # Получаем изображение и метку
    img = img.permute(1, 2, 0)  # Перестановка размерностей для отображения (C, H, W) -> (H, W, C)
    img = img.numpy()  # Преобразуем в NumPy массив
    plt.imshow(img)
    plt.axis('off')
plt.show()