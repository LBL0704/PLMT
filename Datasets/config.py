import ml_collections
from torchvision import transforms
from Datasets.semi_datasets import SemiDataset,RandomGenerator

patch_size = (256, 256)
config_10 = ml_collections.ConfigDict()
config_20 = ml_collections.ConfigDict()
config_all = ml_collections.ConfigDict()
config_val = ml_collections.ConfigDict()
config_only_10 = ml_collections.ConfigDict()
config_only_20 = ml_collections.ConfigDict()


config_10.name = "ACDC"
config_10.root = r"/home/baldwin/PLMT/Datasets/ACDC"
config_10.mode = 'train'
config_10.only_supervision=False
config_10.percent = "10"

config_20.name = "ACDC"
config_20.root = r"/home/baldwin/PLMT/Datasets/ACDC"
config_20.mode = 'train'
config_20.only_supervision=False
config_20.percent = "20"

config_only_10.name = "ACDC"
config_only_10.root = r"/home/baldwin/PLMT/Datasets/ACDC"
config_only_10.mode = 'train'
config_only_10.only_supervision=True
config_only_10.percent = "10"

config_only_20.name = "ACDC"
config_only_20.root = r"/home/baldwin/PLMT/Datasets/ACDC"
config_only_20.mode = 'train'
config_only_20.only_supervision=True
config_only_20.percent = "20"

config_all.name = "ACDC"
config_all.root = r"/home/baldwin/PLMT/Datasets/ACDC"
config_all.mode = 'train'
config_all.only_supervision=False
config_all.percent = "all"

config_val.name = "ACDC"
config_val.root = r"/home/baldwin/PLMT/Datasets/ACDC"
config_val.mode = 'val'
config_val.only_supervision = False
config_val.percent = None

acdc_10_train_dataset = SemiDataset(config_10, transform=transforms.Compose([RandomGenerator(patch_size)]))
acdc_10_pseudo_train_dataset = SemiDataset(config_10,
                                             pseudo_mask=True,
                                             transform=transforms.Compose([RandomGenerator(patch_size, is_mask=True)]))

acdc_20_train_dataset = SemiDataset(config_20, transform=transforms.Compose([RandomGenerator(patch_size)]))
acdc_20_pseudo_train_dataset = SemiDataset(config_20,
                                             pseudo_mask=True,
                                             transform=transforms.Compose([RandomGenerator(patch_size, is_mask=True)]))

acdc_val_dataset = SemiDataset(config_val,
                                   transform=transforms.Compose([RandomGenerator(patch_size, data_aug=False)]))

acdc_unlabeled_dataset = SemiDataset(config_all,
                                       transform=transforms.Compose([RandomGenerator(patch_size, data_aug=False)]))

acdc_all_train_dataset = SemiDataset(config_all, transform=transforms.Compose([RandomGenerator(patch_size)]))

acdc_only_10_train_dataset = SemiDataset(config_only_10, transform=transforms.Compose([RandomGenerator(patch_size)]))
acdc_only_20_train_dataset = SemiDataset(config_only_20, transform=transforms.Compose([RandomGenerator(patch_size)]))

if __name__ == "__main__":
    for i in acdc_all_train_dataset:
        print(i["image"].shape)
        print(i["label"].shape)