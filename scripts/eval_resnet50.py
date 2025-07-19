# eval_resnet50.py
import torch
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, required=True)
parser.add_argument('--ckpt', type=str, required=True)
parser.add_argument('--batch-size', type=int, default=64)
args = parser.parse_args()

# Data loader
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])
val_dataset = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

# Load pruned model
model = torch.load(args.ckpt)
model = model.eval().cuda()

# Evaluate
correct, total = 0, 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.cuda(), labels.cuda()
        outputs = model(images)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

top1_acc = correct / total * 100
print(f"Top-1 Accuracy: {top1_acc:.2f}%")
