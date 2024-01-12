from tqdm import tqdm

#############################################
#                DataLoader                 #
#############################################

## https://github.com/KellerJordan/cifar10-loader/blob/master/quick_cifar/loader.py
import os
from math import ceil
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
#CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616))
CIFAR_STD = torch.tensor([0.2023, 0.1994, 0.2010]) # the robustness repo uses these incorrect std values for cifar

# https://github.com/tysam-code/hlb-CIFAR10/blob/main/main.py#L389
def make_random_square_masks(inputs, size):
    is_even = int(size % 2 == 0)
    n,c,h,w = inputs.shape

    # seed top-left corners of squares to cutout boxes from, in one dimension each
    corner_y = torch.randint(0, h-size+1, size=(n,), device=inputs.device)
    corner_x = torch.randint(0, w-size+1, size=(n,), device=inputs.device)

    # measure distance, using the center as a reference point
    corner_y_dists = torch.arange(h, device=inputs.device).view(1, 1, h, 1) - corner_y.view(-1, 1, 1, 1)
    corner_x_dists = torch.arange(w, device=inputs.device).view(1, 1, 1, w) - corner_x.view(-1, 1, 1, 1)
    
    mask_y = (corner_y_dists >= 0) * (corner_y_dists < size)
    mask_x = (corner_x_dists >= 0) * (corner_x_dists < size)

    final_mask = mask_y * mask_x

    return final_mask

def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)

def batch_crop(inputs, crop_size):
    crop_mask = make_random_square_masks(inputs, crop_size)
    cropped_batch = torch.masked_select(inputs, crop_mask)
    return cropped_batch.view(inputs.shape[0], inputs.shape[1], crop_size, crop_size)

def batch_translate(inputs, translate):
    width = inputs.shape[-2]
    padded_inputs = F.pad(inputs, (translate,)*4, 'constant', value=0)
    return batch_crop(padded_inputs, width)

def batch_cutout(inputs, size):
    cutout_masks = make_random_square_masks(inputs, size)
    return inputs.masked_fill(cutout_masks, 0)

class CifarLoader:

    def __init__(self, path, train=True, batch_size=500, aug=None, drop_last=None, shuffle=None, gpu=0):
        data_path = os.path.join(path, 'train.pt' if train else 'test.pt')
        if not os.path.exists(data_path):
            dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
            images = torch.tensor(dset.data)
            labels = torch.tensor(dset.targets)
            torch.save({'images': images, 'labels': labels, 'classes': dset.classes}, data_path)

        data = torch.load(data_path, map_location=torch.device(gpu))
        self.images, self.labels, self.classes = data['images'], data['labels'], data['classes']
        # It's faster to load+process uint8 data than to load preprocessed fp16 data
        self.images = (self.images.half() / 255).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)

        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)
        self.denormalize = T.Normalize(-CIFAR_MEAN / CIFAR_STD, 1 / CIFAR_STD)
        
        self.aug = aug or {}
        for k in self.aug.keys():
            assert k in ['flip', 'translate', 'cutout'], 'Unrecognized key: %s' % k

        self.batch_size = batch_size
        self.drop_last = train if drop_last is None else drop_last
        self.shuffle = train if shuffle is None else shuffle

    def augment(self, images):
        if self.aug.get('flip', False):
            images = batch_flip_lr(images)
        if self.aug.get('cutout', 0) > 0:
            images = batch_cutout(images, self.aug['cutout'])
        if self.aug.get('translate', 0) > 0:
            # Apply translation in minibatches in order to save memory
            images = torch.cat([batch_translate(image_batch, self.aug['translate'])
                                for image_batch in images.split(5000)])
        return images

    def __len__(self):
        return len(self.images)//self.batch_size if self.drop_last else ceil(len(self.images)/self.batch_size)

    def __iter__(self):
        images = self.augment(self.normalize(self.images))
        indices = (torch.randperm if self.shuffle else torch.arange)(len(images), device=images.device)
        for i in range(len(self)):
            idxs = indices[i*self.batch_size:(i+1)*self.batch_size]
            yield (images[idxs], self.labels[idxs])

def evaluate(model, loader):
    model.eval()
    with torch.no_grad():
        outs = torch.cat([model(inputs) for inputs, _ in loader])
    return (outs.argmax(1) == loader.labels).float().mean().item()

def save_data(loader, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    obj = {'images': loader.images, 'labels': loader.labels}
    torch.save(obj, path)
    
def load_data(loader, path):
    obj = torch.load(path)
    loader.images = obj['images']
    loader.labels = obj['labels']


if __name__ == '__main__':
    from robustness.datasets import CIFAR
    from robustness.model_utils import make_and_restore_model

    #k = '90dbc7ed-e791-44d5-a5a9-fdeb1acfa170' # REPLACE KEY HERE - resnet50 on CIFAR-10 with default settings
    #path = './checkpoints/%s/149_checkpoint.pt' % k
    path = './cifar_nat.pt'
    ds = CIFAR('cifar10')
    model, _ = make_and_restore_model(arch='resnet50', dataset=ds,
                 resume_path=path)
    model.eval()

    test_loader = CifarLoader('/tmp/cifar10', train=False, batch_size=100)
    test_loader.images = test_loader.images.float()
    print('accuracy on clean test set:', evaluate(model.model, test_loader))

    loader = CifarLoader('/tmp/cifar10', train=True, shuffle=False, batch_size=100)
    loader.images = loader.images.float()
    loader.labels = torch.randint_like(loader.labels, low=0, high=10)

    attack_kwargs = {
       'constraint': '2',
       'eps': 0.5,
       'step_size': 0.1,
       'iterations': 100, # Number of PGD steps
       'targeted': True, # Targeted attack
       'custom_loss': None # Use default cross-entropy loss
    }
    adv_images_list = []
    for inputs, targets in tqdm(loader):
        images = loader.denormalize(inputs.contiguous()) # these are properly normalized to [0, 1] range
        assert images.min().abs() < 1e-3
        assert (images.max() - 1).abs() < 1e-3
        _, adv_images = model(images, targets, make_adv=True, **attack_kwargs)
        adv_images_list.append(adv_images)
    adv_images = torch.cat(adv_images_list)

    loader.images = adv_images
    print('targeted attack success rate:', evaluate(model.model, loader))

    save_data(loader, './drand_resnet50_featsnotbugs.pt')

