import kornia.augmentation as K
import torch

from coal_emissions_monitoring.constants import CROP_SIZE_PX, RANDOM_TRANSFORM_PROB


def get_transform(
    data_group: str, crop_size: int = CROP_SIZE_PX
) -> K.AugmentationSequential:
    """
    Get the transform for the given data group, i.e. train, val, or test.

    Args:
        data_group (str): data group
        crop_size (int): crop size

    Returns:
        K.AugmentationSequential: transforms
    """
    if data_group == "train":
        return K.AugmentationSequential(
            K.RandomCrop(size=(crop_size, crop_size)),
            K.RandomHorizontalFlip(p=RANDOM_TRANSFORM_PROB),
            K.RandomRotation(p=RANDOM_TRANSFORM_PROB, degrees=90),
            # TODO this contrast transform is sometimes making the image too dark
            # consider fixing it if needing more regularization
            # K.RandomContrast(p=RANDOM_TRANSFORM_PROB, contrast=(0.9, 1.1)),
            data_keys=["image"],
            same_on_batch=False,
            keepdim=True,
        )
    elif data_group == "val":
        return K.AugmentationSequential(
            K.CenterCrop(size=(crop_size, crop_size)),
            data_keys=["image"],
            same_on_batch=False,
            keepdim=True,
        )
    elif data_group == "test":
        return K.AugmentationSequential(
            K.CenterCrop(size=(crop_size, crop_size)),
            data_keys=["image"],
            same_on_batch=False,
            keepdim=True,
        )
    else:
        raise ValueError(
            f"Invalid data group: {data_group}." "Expected one of: train, val, test."
        )


efficientnet_transform = K.AugmentationSequential(
    K.Resize(size=(256, 256)),
    K.CenterCrop(size=(224, 224)),
    K.Normalize(
        mean=torch.tensor([0.485, 0.456, 0.406]),
        std=torch.tensor([0.229, 0.224, 0.225]),
    ),
    data_keys=["image"],
    same_on_batch=False,
    keepdim=True,
)
