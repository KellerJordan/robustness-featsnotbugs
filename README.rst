robustness package
==================

This is my attempt to replicate [Ilyas et al. (2019), "Adversarial Examples Are Not Bugs, They Are Features"](https://arxiv.org/abs/1905.02175).


Steps:
1. Install via ``pip``: ``pip install robustness``
2. Download the [trained ResNet-50](https://www.dropbox.com/s/yhpp4yws7sgi6lj/cifar_nat.pt?dl=0) provided in the original repo. It should be downloaded as `cifar_nat.pt` in this folder.
3. Run `generate_drand.py` to generate D_rand as in the paper. (takes 20min on A100)
4. Train a new ResNet-50 on the generated D_rand, via the following command (takes ~1hour on A100):

```
python -m robustness.main --dataset cifar --data ./cifar10/ --adv-train 0 --lr 0.01 --arch resnet50 --out-dir ./checkpoints/
```

Note that `robustness/datasets.py` is modified so that cifar points to the newly generated D_rand.


Result: after running this, I'm not seeing any generalization to the clean test set.

---

Other sanity checks I ran:

* Training on the versions of D_rand and D_det from [Ilyas et al.'s data release](https://github.com/MadryLab/constructed-datasets): confirmed that this yields the same generalization as reported in the paper.
* Training a new ResNet-50 from scratch: confirmed that this gets a similar accuracy to their released one, although slightly less because the defaults in this repo are to only train for 150 rather than >=190 epochs. So the training code definitely works.

So the issue seems isolated to my code that generates D_rand. But I can't figure out what could be wrong with it. The attack success rate is over 95%.

