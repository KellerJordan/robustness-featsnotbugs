robustness package
==================

This is my attempt to replicate [Ilyas et al. (2019), "Adversarial Examples Are Not Bugs, They Are Features"](https://arxiv.org/abs/1905.02175).


Steps:
1. Install via `pip`: `pip install robustness`
2. Download the [trained ResNet-50](https://www.dropbox.com/s/yhpp4yws7sgi6lj/cifar_nat.pt?dl=0) provided in the original repo. It should be saved to this folder as `cifar_nat.pt`.
3. Run `generate_drand.py` to generate D_rand, as described in the paper. (takes 20min on A100)
4. Train a new ResNet-50 on the generated D_rand, via the following command (takes ~1hour on A100):

```
python -m robustness.main --dataset cifar --data ./cifar10/ --adv-train 0 --lr 0.01 --arch resnet50 --out-dir ./checkpoints/
```

Note that `robustness/datasets.py` is modified so that cifar points to the newly generated D_rand.


Result: yep, after running this, I do see roughly the same generalization that they report in the paper!

