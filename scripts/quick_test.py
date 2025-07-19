# quick_test.py
import torch
import torch_pruning as tp
from torchvision.models import resnet18

model = resnet18(pretrained=True)
example_inputs = torch.randn(1, 3, 224, 224)
DG = tp.DependencyGraph().build_dependency(model, example_inputs)
print("Torch-Pruning is working!")
