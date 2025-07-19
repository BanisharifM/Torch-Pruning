import torch_pruning as tp
import torchvision.models as models
import torch, argparse, os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def main(args):
    # Dataset
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    val_dataset = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=16, pin_memory=True)

    # Load model
    model = models.resnet50(pretrained=True).eval().cuda()
    example_inputs = torch.randn(1,3,224,224).cuda()

    # Pruning importance
    if args.pruning_method == 'l1':
        importance = tp.importance.MagnitudeImportance(p=1)
    elif args.pruning_method == 'l2':
        importance = tp.importance.MagnitudeImportance(p=2)
    elif args.pruning_method == 'random':
        importance = tp.importance.RandomImportance()
    else:
        raise ValueError("Unsupported pruning method: {}".format(args.pruning_method))

    # Apply pruning
    pruner = tp.pruner.MagnitudePruner(
        model, example_inputs,
        importance=importance,
        iterative_steps=1,
        ch_sparsity=args.pruning_ratio,
    )
    pruner.step()

    # Save pruned model
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"resnet50_{args.pruning_method}.pth")
    torch.save(model, save_path)
    print(f"Pruned model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--pruning-method', type=str, choices=['l1', 'l2', 'random'], default='l1')
    parser.add_argument('--pruning-ratio', type=float, default=0.5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--save-dir', type=str, default='output/pruned')
    args = parser.parse_args()
    main(args)
