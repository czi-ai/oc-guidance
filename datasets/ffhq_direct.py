# Implements dataset and dataloader that consume FFHQ .png files directly,
# instead of via lmdb files like the original datasets/ffhq.py.
import os
import torch
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms


class FFHQDirectDataset(data.Dataset):
    def __init__(
        self,
        root,
        split="train",
        transform=None,
        image_size=256,
        subset=-1,
        name=None,
        **kwargs,
    ):
        """
        Args:
            root (str): Path to the FFHQ images directory
            split (str): 'train' or 'val'
            transform (callable, optional): Optional transform to be applied on a sample
            image_size (int): Target image size
            subset (int or list): If int > 0, use first N images. If list, use those indices.
            name (str, optional): Dataset name (ignored)
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.image_size = image_size

        # Load train/val split indices
        ind_path = os.path.join(root, 'train_test_ind.pt')
        if os.path.exists(ind_path):
            ind_dat = torch.load(ind_path, weights_only=False)
            train_ind = ind_dat['train']
            test_ind = ind_dat['test']
        else:
            # If no split file exists, create one
            num_images = 70000
            num_train = 63000
            rand = np.random.permutation(num_images)
            train_ind = rand[:num_train]
            test_ind = rand[num_train:]
            torch.save({'train': train_ind, 'test': test_ind}, ind_path)

        # Select indices based on split
        self.indices = train_ind if split == 'train' else test_ind

        # Apply subset if specified
        if isinstance(subset, int) and subset > 0:
            self.indices = self.indices[:subset]
        elif isinstance(subset, list):
            self.indices = subset

    def __getitem__(self, index):
        target = 0
        img_idx = self.indices[index]
        img_path = os.path.join(self.root, f'{img_idx:05d}.png')
        
        # Load and resize image
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, target, {"index": index}

    def __len__(self):
        return len(self.indices)


def get_ffhq_dataset(root, split, transform="default", subset=-1, **kwargs):
    """
    Get FFHQ dataset that loads images directly from disk.
    
    Args:
        root (str): Path to the FFHQ images directory
        split (str): 'train' or 'val'
        transform (str): Type of transform to apply
        subset (int or list): If int > 0, use first N images. If list, use those indices.
    """
    if transform == "default":
        transform = transforms.ToTensor()
    elif transform == "identity":
        transform = transforms.Compose([transforms.PILToTensor()])
    else:
        raise ValueError(f"Unsupported transform: {transform}")

    dset = FFHQDirectDataset(root=root, split=split, transform=transform, **kwargs)
    if isinstance(subset, int) and subset > 0:
        dset = data.Subset(dset, list(range(subset)))
    elif isinstance(subset, list):
        dset = data.Subset(dset, subset)
    return dset


def get_ffhq_loader(
    dset,
    *,
    batch_size,
    num_workers,
    shuffle,
    drop_last,
    pin_memory,
    **kwargs
):
    """
    Get DataLoader for FFHQ dataset.
    """
    sampler = data.distributed.DistributedSampler(dset, shuffle=shuffle, drop_last=drop_last)
    loader = data.DataLoader(
        dset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=pin_memory,
        # persistent_workers=True,
        # persistent_workers=(num_workers > 0)
        persistent_workers=False
    )
    return loader 
