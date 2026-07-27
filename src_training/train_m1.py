import os
import torch
import torch.utils.data
import torchvision
import xml.etree.ElementTree as ET
import numpy as np
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


# ----------------- CONFIGURAZIONE -----------------
class Config:
    # Percorso adattato alla nostra struttura di cartelle
    DATA_ROOT = '../data/neu-det'
    CLASSES = ['__background__', 'crazing', 'inclusion', 'patches',
               'pitted_surface', 'rolled-in_scale', 'scratches']
    NUM_CLASSES = len(CLASSES)
    BATCH_SIZE = 8  # Ridotto da 16 per evitare saturazione RAM unificata del Mac
    NUM_EPOCHS = 10  # Abbassato per fare un test veloce, poi potrai alzarlo a 20
    LEARNING_RATE = 0.005
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    STEP_SIZE = 5
    GAMMA = 0.1
    NUM_WORKERS = 2

    # Rilevamento automatico Apple M1 (MPS)
    if torch.backends.mps.is_available():
        DEVICE = torch.device('mps')
    elif torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')


# ----------------- DATASET -----------------
class NEUDETDataset(Dataset):
    def __init__(self, root_dir, split='train', transforms=None):
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms
        self.imgs_dir = os.path.join(root_dir, split, 'images')
        self.annotations_dir = os.path.join(root_dir, split, 'annotations')
        self.annotations = sorted([f for f in os.listdir(self.annotations_dir) if f.endswith('.xml')])
        self.class_to_idx = {name: idx for idx, name in enumerate(Config.CLASSES)}

    def __len__(self):
        return len(self.annotations)

    def parse_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes, labels = [], []
        filename = root.find('filename').text
        for obj in root.findall('object'):
            label = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[label])
        return filename, boxes, labels

    def find_image_path(self, filename):
        possible_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'] if not os.path.splitext(filename)[
            1] else ['']
        for category in ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']:
            for ext in possible_extensions:
                img_path = os.path.join(self.imgs_dir, category, filename + ext)
                if os.path.exists(img_path): return img_path
        for ext in possible_extensions:
            img_path = os.path.join(self.imgs_dir, filename + ext)
            if os.path.exists(img_path): return img_path
        raise FileNotFoundError(f"Immagine {filename} non trovata.")

    def __getitem__(self, idx):
        ann_path = os.path.join(self.annotations_dir, self.annotations[idx])
        filename, boxes, labels = self.parse_xml(ann_path)
        img_path = self.find_image_path(filename)
        img = Image.open(img_path).convert('RGB')

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)
        image_id = torch.tensor([idx])

        target = {'boxes': boxes, 'labels': labels, 'image_id': image_id, 'area': area, 'iscrowd': iscrowd}
        if self.transforms: img = self.transforms(img)
        return img, target


def get_transform(train=False):
    transforms = [T.ToTensor()]
    if train: transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def collate_fn(batch):
    return tuple(zip(*batch))


# ----------------- MODELLO E LOOP -----------------
def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc=f'Epoch {epoch}')

    for images, targets in progress_bar:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        progress_bar.set_postfix(
            {'loss': f'{losses.item():.4f}', 'avg_loss': f'{total_loss / (progress_bar.n + 1):.4f}'})
    return total_loss / len(data_loader)


@torch.no_grad()
def evaluate_with_metrics(model, data_loader, device):
    model.eval()
    metric = MeanAveragePrecision()
    for images, targets in tqdm(data_loader, desc='Validation'):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        predictions = model(images)
        preds = [{k: v.cpu() for k, v in pred.items()} for pred in predictions]
        targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]
        metric.update(preds, targets_cpu)
    return metric.compute()


def main():
    print(f"[*] Hardware in uso: {Config.DEVICE}")
    train_dataset = NEUDETDataset(Config.DATA_ROOT, split='train', transforms=get_transform(train=True))
    val_dataset = NEUDETDataset(Config.DATA_ROOT, split='val', transforms=get_transform(train=False))

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS,
                            collate_fn=collate_fn)

    model = get_model(Config.NUM_CLASSES).to(Config.DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=Config.LEARNING_RATE, momentum=Config.MOMENTUM,
                                weight_decay=Config.WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=Config.STEP_SIZE, gamma=Config.GAMMA)

    best_map = 0.0
    model_save_path = "../models/best_model.pth"

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, optimizer, train_loader, Config.DEVICE, epoch)
        metrics = evaluate_with_metrics(model, val_loader, Config.DEVICE)
        map_50 = metrics['map_50'].item()
        print(f"[*] Validation mAP@50: {map_50:.4f} | Loss: {train_loss:.4f}")

        lr_scheduler.step()

        if map_50 > best_map:
            best_map = map_50
            # Salva i pesi nella cartella corretta per i nostri benchmark
            torch.save({'model_state_dict': model.state_dict()}, model_save_path)
            print(f"[+] Salvato nuovo best_model in: {model_save_path}")


if __name__ == '__main__':
    # Assicurati che la cartella models esista
    os.makedirs("../models", exist_ok=True)
    main()