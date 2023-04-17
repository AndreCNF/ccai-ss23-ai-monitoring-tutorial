import kornia.augmentation as K

from coal_emissions_monitoring.constants import CROP_SIZE_PX, RANDOM_TRANSFORM_PROB

train_transforms = K.AugmentationSequential(
    K.RandomCrop(size=(CROP_SIZE_PX, CROP_SIZE_PX)),
    K.RandomHorizontalFlip(p=RANDOM_TRANSFORM_PROB),
    K.RandomRotation(p=RANDOM_TRANSFORM_PROB, degrees=90),
    # TODO this contrast transform is sometimes making the image too dark
    # consider fixing it if needing more regularization
    # K.RandomContrast(p=RANDOM_TRANSFORM_PROB, contrast=(0.9, 1.1)),
    data_keys=["image"],
    same_on_batch=False,
    keepdim=True,
)

val_transforms = K.AugmentationSequential(
    K.CenterCrop(size=(CROP_SIZE_PX, CROP_SIZE_PX)),
    data_keys=["image"],
    same_on_batch=False,
    keepdim=True,
)

test_transforms = val_transforms
