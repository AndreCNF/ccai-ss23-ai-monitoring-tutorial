from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from lightning import LightningDataModule
import geopandas as gpd
from tqdm.auto import tqdm

from coal_emissions_monitoring.constants import (
    BATCH_SIZE,
    CROP_SIZE_PX,
    EMISSIONS_TARGET,
    MAIN_COLUMNS,
    IMAGE_SIZE_PX,
    MAX_BRIGHT_MEAN,
    MAX_CLOUD_COVER_PRCT,
    MAX_DARK_FRAC,
    TEST_YEAR,
    TRAIN_VAL_RATIO,
)
from coal_emissions_monitoring.satellite_imagery import (
    fetch_image_path_from_cog,
    get_image_from_cog,
    is_image_too_bright,
    is_image_too_dark,
)
from coal_emissions_monitoring.data_cleaning import (
    filter_to_cooling_tower_plants,
    get_final_dataset,
    load_final_dataset,
)
from coal_emissions_monitoring.ml_utils import (
    get_facility_set_mapper,
    split_data_in_sets,
)
from coal_emissions_monitoring.transforms import get_transform


class CoalEmissionsDataset(IterableDataset):
    def __init__(
        self,
        gdf: gpd.GeoDataFrame,
        target: str = EMISSIONS_TARGET,
        image_size: int = IMAGE_SIZE_PX,
        max_dark_frac: float = MAX_DARK_FRAC,
        max_mean_val: float = MAX_BRIGHT_MEAN,
        transforms: Optional[torch.nn.Module] = None,
        use_local_images: bool = False,
    ):
        """
        Dataset that gets images of coal power plants, their emissions
        and metadata.

        Args:
            gdf (gpd.GeoDataFrame):
                A GeoDataFrame with the following columns:
                - facility_id
                - facility_name
                - latitude
                - longitude
                - ts
                - co2_mass_short_tons
                - cloud_cover
                - cog_url
                - geometry
            target (str):
                The target column to predict
            image_size (int):
                The size of the image in pixels
            max_dark_frac (float):
                The maximum fraction of dark pixels allowed for an image;
                if the image has more dark pixels than this, it is skipped
            max_mean_val (float):
                The maximum mean value allowed for an image; if the image
                has a higher mean value than this, it is skipped
            transforms (Optional[torch.nn.Module]):
                A PyTorch module that transforms the image
            use_local_images (bool):
                Whether to use local images instead of downloading them
                from the cloud
        """
        assert len(set(MAIN_COLUMNS) - set(gdf.columns)) == 0, (
            "gdf must have all columns of the following list:\n"
            f"{MAIN_COLUMNS}\n"
            f"Instead, gdf has the following columns:\n"
            f"{gdf.columns}"
        )
        self.gdf = gdf
        self.target = target
        self.image_size = image_size
        self.max_dark_frac = max_dark_frac
        self.max_mean_val = max_mean_val
        self.transforms = transforms
        self.use_local_images = use_local_images
        if self.use_local_images:
            assert "local_image_path" in self.gdf.columns, (
                "If use_local_images is True, gdf must have a "
                "local_image_path column"
            )

    def __iter__(self):
        if torch.utils.data.get_worker_info():
            worker_total_num = torch.utils.data.get_worker_info().num_workers
            worker_id = torch.utils.data.get_worker_info().id
        else:
            worker_total_num = 1
            worker_id = 0
        for idx in range(worker_id, len(self.gdf), worker_total_num):
            row = self.gdf.iloc[idx]
            if self.use_local_images:
                image = np.load(row.local_image_path)
            else:
                image = get_image_from_cog(
                    cog_url=row.cog_url, geometry=row.geometry, size=self.image_size
                )
            image = torch.from_numpy(image).float()
            if is_image_too_dark(
                image, max_dark_frac=self.max_dark_frac
            ) or is_image_too_bright(image, max_mean_val=self.max_mean_val):
                continue
            if self.transforms is not None:
                image = self.transforms(image).squeeze(0)
            target = torch.tensor(row[self.target]).float()
            metadata = row.drop([self.target, "geometry", "data_set"]).to_dict()
            metadata["ts"] = str(metadata["ts"])
            yield {
                "image": image,
                "target": target,
                "metadata": metadata,
            }


class CoalEmissionsDataModule(LightningDataModule):
    def __init__(
        self,
        final_dataset_path: Optional[Union[str, Path]] = None,
        image_metadata_path: Optional[Union[str, Path]] = None,
        campd_facilities_path: Optional[Union[str, Path]] = None,
        campd_emissions_path: Optional[Union[str, Path]] = None,
        target: str = EMISSIONS_TARGET,
        image_size: int = IMAGE_SIZE_PX,
        crop_size: int = CROP_SIZE_PX,
        train_val_ratio: float = TRAIN_VAL_RATIO,
        test_year: int = TEST_YEAR,
        batch_size: int = BATCH_SIZE,
        max_dark_frac: float = MAX_DARK_FRAC,
        max_mean_val: float = MAX_BRIGHT_MEAN,
        max_cloud_cover_prct: int = MAX_CLOUD_COVER_PRCT,
        predownload_images: bool = False,
        download_missing_images: bool = False,
        images_dir: str = "images/",
        must_have_cooling_tower: bool = False,
        num_workers: int = 0,
    ):
        """
        Lightning Data Module that gets images of coal power plants,
        their emissions and metadata, and splits them into train,
        validation and test sets.

        Args:
            image_metadata_path (Union[str, Path]):
                Path to image metadata data
            campd_facilities_path (Union[str, Path]):
                Path to CAMPD facilities data
            campd_emissions_path (Union[str, Path]):
                Path to CAMPD emissions data
            target (str):
                The target column to predict
            image_size (int):
                The size of the image in pixels
            crop_size (int):
                The size of the crop in pixels
            train_val_ratio (float):
                The ratio of train to validation data
            test_year (int):
                The year to use for testing
            batch_size (int):
                The batch size, i.e. the number of samples to load at once
            max_dark_frac (float):
                The maximum fraction of dark pixels allowed for an image;
                if the image has more dark pixels than this, it is skipped
            max_mean_val (float):
                The maximum mean value allowed for an image; if the image
                has a higher mean value than this, it is skipped
            max_cloud_cover_prct (int):
                The maximum cloud cover percentage allowed for an image;
                if the image has more cloud cover than this, it is skipped
            predownload_images (bool):
                Whether to pre-download images from the cloud or load each
                one on the fly
            download_missing_images (bool):
                Whether to download images that are missing from the
                images_dir path
            images_dir (str):
                The directory to save images to if predownload_images is True
            must_have_cooling_tower (bool):
                Whether to only include images of power plants that have
                cooling tower(s)
            num_workers (int):
                The number of workers to use for loading data
        """
        super().__init__()
        self.final_dataset_path = final_dataset_path
        self.image_metadata_path = image_metadata_path
        self.campd_facilities_path = campd_facilities_path
        self.campd_emissions_path = campd_emissions_path
        self.target = target
        self.image_size = image_size
        self.crop_size = crop_size
        self.train_val_ratio = train_val_ratio
        self.test_year = test_year
        self.batch_size = batch_size
        self.max_dark_frac = max_dark_frac
        self.max_mean_val = max_mean_val
        self.max_cloud_cover_prct = max_cloud_cover_prct
        self.predownload_images = predownload_images
        self.download_missing_images = download_missing_images
        self.images_dir = images_dir
        self.must_have_cooling_tower = must_have_cooling_tower
        self.num_workers = num_workers
        self.emissions_quantiles = None

    def setup(self, stage: str):
        """
        Split the data into train, validation and test sets.

        Args:
            stage (str):
                The stage of the setup
        """
        # load the final dataset
        if self.final_dataset_path is not None:
            self.gdf = load_final_dataset(self.final_dataset_path)
        else:
            self.gdf = get_final_dataset(
                image_metadata_path=self.image_metadata_path,
                campd_facilities_path=self.campd_facilities_path,
                campd_emissions_path=self.campd_emissions_path,
            )
        # filter out rows with too much cloud cover
        self.gdf = self.gdf[self.gdf.cloud_cover <= self.max_cloud_cover_prct]
        if self.predownload_images:
            # make sure that images are already downloaded
            if "local_image_path" not in self.gdf.columns:
                tqdm.pandas(desc="Downloading images")
                self.gdf["local_image_path"] = self.gdf.progress_apply(
                    lambda row: fetch_image_path_from_cog(
                        cog_url=row.cog_url,
                        geometry=row.geometry,
                        size=self.image_size,
                        images_dir=self.images_dir,
                        download_missing_images=self.download_missing_images,
                    ),
                    axis=1,
                )
                # skip rows where the image could not be downloaded
                self.gdf = self.gdf[~self.gdf.local_image_path.isna()]
            else:
                # make sure that the image paths are in the right directory
                current_image_path = (
                    self.gdf.local_image_path.str.split("/")
                    .str[:-1]
                    .str.join("/")
                    .iloc[0]
                )
                if current_image_path != self.images_dir:
                    self.gdf.local_image_path = self.gdf.local_image_path.str.replace(
                        current_image_path, self.images_dir
                    )
        if self.must_have_cooling_tower:
            self.gdf = filter_to_cooling_tower_plants(
                gdf=self.gdf,
                campd_facilities_path=self.campd_facilities_path,
            )
        # split the data into train, validation and test sets
        facility_set_mapper = get_facility_set_mapper(
            campd_facilities_path=self.campd_facilities_path,
            train_val_ratio=self.train_val_ratio,
        )
        self.gdf["data_set"] = self.gdf.apply(
            lambda row: split_data_in_sets(
                row=row, data_set_mapper=facility_set_mapper, test_year=self.test_year
            ),
            axis=1,
        )
        if stage == "fit":
            self.train_dataset = CoalEmissionsDataset(
                gdf=self.gdf[self.gdf.data_set == "train"].sample(frac=1),
                target=self.target,
                image_size=self.image_size,
                transforms=get_transform(data_group="train", crop_size=self.crop_size),
                use_local_images=self.predownload_images,
                max_dark_frac=self.max_dark_frac,
                max_mean_val=self.max_mean_val,
            )
            self.emissions_quantiles = self.calculate_emissions_quantiles()
            self.val_dataset = CoalEmissionsDataset(
                gdf=self.gdf[self.gdf.data_set == "val"].sample(frac=1),
                target=self.target,
                image_size=self.image_size,
                transforms=get_transform(data_group="val", crop_size=self.crop_size),
                use_local_images=self.predownload_images,
                max_dark_frac=self.max_dark_frac,
                max_mean_val=self.max_mean_val,
            )
        elif stage == "test":
            self.test_dataset = CoalEmissionsDataset(
                gdf=self.gdf[self.gdf.data_set == "test"].sample(frac=1),
                target=self.target,
                image_size=self.image_size,
                transforms=get_transform(data_group="test", crop_size=self.crop_size),
                use_local_images=self.predownload_images,
                max_dark_frac=self.max_dark_frac,
                max_mean_val=self.max_mean_val,
            )

    def get_dataloader(self, data_group: str):
        # reshuffle the dataset
        getattr(self, f"{data_group}_dataset").gdf = getattr(
            self, f"{data_group}_dataset"
        ).gdf.sample(frac=1)
        # reset the dataloader
        return DataLoader(
            getattr(self, f"{data_group}_dataset"),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
        )

    def train_dataloader(self):
        return self.get_dataloader("train")

    def val_dataloader(self):
        return self.get_dataloader("val")

    def test_dataloader(self):
        return self.get_dataloader("test")

    def calculate_emissions_quantiles(self) -> Dict[float, float]:
        """
        Calculate the quantiles of the emissions for the train set.

        Returns:
            Dict[float, float]:
                The quantiles of the emissions for the train set.
        """
        return self.train_dataset.gdf[self.target].quantile([0.3, 0.6, 0.95]).to_dict()
