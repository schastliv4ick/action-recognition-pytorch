import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from models.__all_models import PoseCNNsc

transform = transforms.Compose([
    transforms.Resize((288, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_name

test_dir = 'C:\\Users\\Semyon\\YandexLyceum\\project\\yandex-ml-2025\\data\\human_poses_data\\img_test'
dataset = TestDataset(image_dir=test_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

num_classes = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PoseCNNsc(num_classes=num_classes)
model.load_state_dict(torch.load(
    'C:\\Users\\Semyon\\YandexLyceum\\project\\yandex-ml-2025\\src\\saves\\PoseCNNsc_best_f1.pt',
    map_location=device
))
model.to(device)
model.eval()


all_preds = []
all_img_names = []

with torch.no_grad():
    for images, img_names in tqdm(dataloader):
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().tolist())
        all_img_names.extend(img_names)


results_df = pd.DataFrame({
    'img_name': all_img_names,
    'predicted': all_preds 
})
results_df.to_csv('inference_results.csv', index=False)
