import matplotlib.pyplot as plt
import torch  # For checking tensor type
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
    transforms.RandomRotation(degrees=15),
    
    # Случайное изменение размера и обрезка
    transforms.RandomResizedCrop(size=(288, 512), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
    
    # Случайное изменение цветовых характеристик
    RandomAdjustColor(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    
    # Преобразование в тензор PyTorch
    transforms.ToTensor(),
    
    # Нормализация изображения (стандартные значения для ImageNet)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform1 = transforms.Compose([
    # 1. Повороты: Поворачивайте изображение на случайный угол в диапазоне [-30°, +30°]
    transforms.RandomRotation(degrees=(-30, 30)),
    
    # 2. Отражение по горизонтали: Отразите изображение по горизонтали (flip)
    transforms.RandomHorizontalFlip(p=0.5),
    
    # 3. Случайное масштабирование: Изменяйте масштаб изображения в пределах [0.8, 1.2]
    transforms.Resize((224, 224)),
    
    # 4. Сдвиги по осям X и Y: Применяем случайные сдвиги
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    
    # 5. Изменение яркости: Случайно изменяйте яркость изображения в диапазоне [0.7, 1.3]
    transforms.ColorJitter(brightness=(0.7, 1.3)),
    
    # 6. Изменение контраста: Меняйте контраст изображения в диапазоне [0.8, 1.2]
    transforms.ColorJitter(contrast=(0.8, 1.2)),
    
    # 7. Изменение насыщенности цветов: Регулируйте насыщенность в диапазоне [0.5, 1.5]
    transforms.ColorJitter(saturation=(0.5, 1.5)),
    
    # Преобразование в тензор PyTorch (должно быть перед Lambda)
    transforms.ToTensor(),
# 8. Добавление шума: Добавляйте небольшое количество случайного шума (например, гауссовский шум)
    transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05),
    
    # 9. Перспективные искажения: Применяйте случайные перспективные преобразования
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    
    # 10. Кроппинг (обрезка): Обрезайте случайную часть изображения, оставляя ключевые области
    transforms.RandomCrop(size=(224, 224), padding=10),
    
    # 11. Размытие: Применяйте случайное размытие (например, Gaussian blur)
    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
    
    # 12. Изменение качества изображения: Уменьшайте разрешение или добавляйте JPEG-артефакты
    transforms.RandomAdjustSharpness(sharpness_factor=0.5, p=0.5),
    
    # 13. Затемнение или затемненные области: Добавляйте затемненные участки ("тени")
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
    
    # Нормализация изображения (необходимо для большинства нейронных сетей)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Путь к данным
# PATH_TO_DATA = "D:\\VS\\ml\\human_poses_data\\yandex-ml-2025\\data"


"""
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
"""