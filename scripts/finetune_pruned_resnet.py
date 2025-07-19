# finetune_pruned_resnet.py
import argparse, os, torch, torch.nn as nn
import torch.distributed as dist
import torchvision
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])
    train_set = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform)
    sampler = DistributedSampler(train_set)
    train_loader = DataLoader(train_set, batch_size=args.batch_size // dist.get_world_size(),
                              sampler=sampler, num_workers=8, pin_memory=True)

    model = torch.load(args.model_path, map_location='cpu')
    model.cuda()
    model = DDP(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.epochs):
        model.train()
        sampler.set_epoch(epoch)
        for images, targets in train_loader:
            images, targets = images.cuda(), targets.cuda()
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        if rank == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}.pth")
            torch.save(model.module.state_dict(), ckpt_path)

if __name__ == "__main__":
    main()
