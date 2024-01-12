# Train a ResNet50 on CIFAR-10 with default settings
python -m robustness.main --dataset cifar --data ./cifar10/ --adv-train 0 --arch resnet50 --out-dir ./checkpoints/
