# 先不管数据增强，弄出来的时候再管数据增强

import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
from torchvision import transforms
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler

class SemiDataset(Dataset):

    def __init__(self, config, pseudo_mask=False, transform=None):
        super(SemiDataset, self).__init__()
        self.name = config.name
        self.root = config.root
        self.mode = config.mode
        self.percent = config.percent
        self.only_supervison = config.only_supervision
        self.pseudo_mask = pseudo_mask
        self.transform = transform

        # self.mode的模式只有两种，train or val
        if self.mode == "val":
            file_val = open(os.path.join(self.root, "test.txt"))
            self.ids = [name[:-1] for name in file_val.readlines()]
        elif self.mode == "train":
            unlabeled_sample_name = []
            if self.percent == "10":
                file_all_samples = open(os.path.join(self.root, "train_all_samples.txt"))
                file_labeled_sample = open(os.path.join(self.root, "train_10_samples.txt"))
                all_train_name = [name[:-1] for name in file_all_samples.readlines()]
                labeled_sample_name = [name[:-1] for name in file_labeled_sample.readlines()]
                self.labeled_samples = labeled_sample_name
                if not self.only_supervison:
                    for name in all_train_name:
                        if name not in labeled_sample_name:
                            unlabeled_sample_name.append(name)
                    self.ids = labeled_sample_name + unlabeled_sample_name
                else:
                    self.ids = labeled_sample_name
            elif self.percent == "20":
                file_all_samples = open(os.path.join(self.root, "train_all_samples.txt"))
                file_labeled_sample = open(os.path.join(self.root, "train_20_samples.txt"))
                all_train_name = [name[:-1] for name in file_all_samples.readlines()]
                labeled_sample_name = [name[:-1] for name in file_labeled_sample.readlines()]
                self.labeled_samples = labeled_sample_name
                if not self.only_supervison:
                    for name in all_train_name:
                        if name not in labeled_sample_name:
                            unlabeled_sample_name.append(name)
                    self.ids = labeled_sample_name + unlabeled_sample_name
                else:
                    self.ids = labeled_sample_name
            elif self.percent == "all":
                file_all_samples = open(os.path.join(self.root, "train_all_samples.txt"))
                self.ids = [name[:-1] for name in file_all_samples.readlines()]
            else:
                assert "wrong training set"
        else:
            assert "wrong mode set"


    def __getitem__(self, item):
        # 这个地方切记啊，self.ids是有顺序的
        case = self.ids[item]
        if self.mode == "val":
            h5f = h5py.File(self.root + r"/test/{}.h5".format(case), "r")
            image = h5f["image"][:]
            label = h5f["label"][:]
            sample = {"image": image, "label": label}
            sample["id"] = case
            return sample

        else:
            if not self.pseudo_mask:
                h5f = h5py.File(self.root + r"/source/{}".format(case), "r")
                image = h5f["image"][:]
                label = h5f["label"][:]
                sample = {"image": image, "label": label}
                if self.transform is not None:
                    sample = self.transform(sample)
                sample["id"] = case
                return sample
            else:
                # 这里的数据增强怎么使用还是要再次思考一下
                h5f_1 = h5py.File(self.root + r"/source/{}".format(case), "r")
                image = h5f_1["image"][:]
                label = h5f_1["label"][:]
                sample = {"image": image, "label": label}
                h5f_2 = h5py.File(self.root + r"/pseudo/{}".format(case), "r")
                pseudo = h5f_2["pseudo"][:]
                mask = h5f_2["mask"][:]
                sample["pseudo"] = pseudo
                mask = mask.astype(np.uint8)
                sample["mask"] = mask

                if case in self.labeled_samples:
                    transform_sample = {"image": image, "label": label, "mask":mask}
                else:
                    transform_sample = {"image": image, "label": pseudo, "mask": mask}
                if self.transform is not None:
                    sample = self.transform(transform_sample)
                sample["id"] = case
                return sample

    def __len__(self):
        return len(self.ids)


def random_rot_flip(image, label, mask=None, is_mask=False):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.rot90(label, k)
    label = np.flip(label, axis=axis).copy()
    if is_mask and mask is not None:
        mask = np.rot90(mask, k)
        mask = np.flip(mask, axis=axis).copy()
        return image, label, mask
    else:
        return image, label



def random_rotate(image, label, mask=None, is_mask=False):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    if is_mask and mask is not None:
        mask = ndimage.rotate(mask, angle, order=0, reshape=False)
        return image, label, mask
    else:
        return image, label


def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)

    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter(image)

def data_aug(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)
    rand_gray_scale = transforms.RandomGrayscale(p=0.2)
    kernel_size = int(random.random() * 4.95)
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
    blurring_image = transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))

    strong_aug = image
    if strong_aug.shape[0] == 1:
        strong_aug = strong_aug.repeat(3, 1, 1)
    if random.random() < 0.8:
        strong_aug = color_jitter(strong_aug)
    strong_aug = rand_gray_scale(strong_aug)

    if random.random() < 0.5:
        strong_aug = blurring_image(strong_aug)
    strong_aug = ((strong_aug[0] + strong_aug[1] + strong_aug[2]) / 3).unsqueeze(0)
    return strong_aug


class RandomGenerator(object):
    def __init__(self, output_size, data_aug=True, is_mask=False):
        self.output_size = output_size
        self.is_mask = is_mask
        self.data_aug=data_aug

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        if self.is_mask:
            mask = sample["mask"]
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if self.data_aug:
            if self.is_mask:
                if random.random() > 0.5:
                    image, label, mask = random_rot_flip(image, label, mask, is_mask=self.is_mask)
                elif random.random() > 0.5:
                    image, label, mask = random_rotate(image, label, mask, is_mask=self.is_mask)
            else:
                if random.random() > 0.5:
                    image, label = random_rot_flip(image, label)
                elif random.random() > 0.5:
                    image, label = random_rotate(image, label)
            img_x, img_y = image.shape
            lab_x, lab_y = label.shape
            image = zoom(image, (self.output_size[0] / img_x, self.output_size[1] / img_y), order=0)
            label = zoom(label, (self.output_size[0] / lab_x, self.output_size[1] / lab_y), order=0)
            if self.is_mask:
                mak_x,mak_y = mask.shape
                mask = zoom(mask, (self.output_size[0] / mak_x, self.output_size[1] / mak_y), order=0)

            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
            label = torch.from_numpy(label.astype(np.uint8))
            if self.is_mask:
                mask = torch.from_numpy(mask.astype(np.bool))
                sample = {"image": image, "label": label, "mask": mask}
            else:
                sample = {"image": image, "label": label}
            return sample
        else:
            img_x, img_y = image.shape
            lab_x, lab_y = label.shape
            image = zoom(image, (self.output_size[0] / img_x, self.output_size[1] / img_y), order=0)
            label = zoom(label, (self.output_size[0] / lab_x, self.output_size[1] / lab_y), order=0)
            if self.is_mask:
                mak_x,mak_y = mask.shape
                mask = zoom(mask, (self.output_size[0] / mak_x, self.output_size[1] / mak_y), order=0)

            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
            label = torch.from_numpy(label.astype(np.uint8))
            if self.is_mask:
                mask = torch.from_numpy(mask.astype(np.bool))
                sample = {"image": image, "label": label, "mask": mask}
            else:
                sample = {"image": image, "label": label}
            return sample



class WeakStrongAugment(object):
    """returns weakly and strongly augmented images

    Args:
        object (tuple): output size of network
    """

    def __init__(self, output_size, is_mask=False):
        self.output_size = output_size
        self.is_mask = is_mask

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        if self.is_mask:
            mask = sample["mask"]
        image = self.resize(image)
        label = self.resize(label)
        if self.is_mask:
            mask = self.resize(mask)
        # weak augmentation is rotation / flip
        if random.random() > 0.5:
            if self.is_mask:
                image, label, mask = random_rot_flip(image, label, is_mask=self.is_mask)
            else:
                image, label = random_rot_flip(image, label)
        if random.random() > 0.5:
            if self.is_mask:
                image, label, mask = random_rotate(image, label, is_mask=self.is_mask)
            else:
                image, label = random_rotate(image, label)
        # strong augmentation is color jitter
        image_strong = data_aug(image).type("torch.FloatTensor")
        # fix dimensions

        image_weak = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        if self.is_mask:
            sample = {
                "image_weak": image_weak,
                "image_strong": image_strong,
                "label_aug": label,
                "mask":mask}
        else:
            sample = {
                "image_weak": image_weak,
                "image_strong": image_strong,
                "label_aug": label}
        return sample

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)



class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


if __name__ == "__main__":
    import ml_collections
    config = ml_collections.ConfigDict()
    config.name = "LA2018"
    config.root =  r"/home/baldwin/PLMT/Datasets/ACDC"
    config.mode = 'train'
    config.only_supervision = "False"
    config.percent = "10"
    patch_size = (256, 256)
    dataset = SemiDataset(config, transform=transforms.Compose([RandomGenerator(patch_size)]))
    count = 1
    for i in dataset:
        print(count, i['id'])
        count += 1

