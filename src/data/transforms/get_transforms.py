from torchvision import transforms


NORMALIZATION_DICT = {
    'imagenet': dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
}

def get_transforms(args):

    transform_args = args.DATA.TRANSFORMS
    
    if transform_args.AUG_TYPE == 'imagenet':

        train_transform = transforms.Compose([
            transforms.Resize(int(transform_args.IMAGE_SIZE / transform_args.CROP_PCT), transform_args.INTERPOLATION),
            transforms.RandomCrop(transform_args.IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize(**NORMALIZATION_DICT[transform_args.NORM_TYPE])
        ])

        test_transform = transforms.Compose([
            transforms.Resize(int(transform_args.IMAGE_SIZE / transform_args.CROP_PCT), transform_args.INTERPOLATION),
            transforms.CenterCrop(transform_args.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(**NORMALIZATION_DICT[transform_args.NORM_TYPE])
        ])

    else:
        raise ValueError(f"Unknown AUG_TYPE: {transform_args.AUG_TYPE}")
    
    return train_transform, test_transform