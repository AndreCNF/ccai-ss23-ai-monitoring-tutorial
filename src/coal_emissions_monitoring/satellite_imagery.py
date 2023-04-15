from datetime import datetime
from typing import Optional

import geopandas as gpd
import numpy as np
import rasterio as rio
import pandas as pd
import pystac_client
from loguru import logger
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from shapely.geometry.base import BaseGeometry
from tqdm.auto import tqdm

from coal_emissions_monitoring.constants import (
    AOI_SIZE_METERS,
    API_URL,
    COLLECTION,
    END_DATE,
    GLOBAL_EPSG,
    IMAGE_SIZE_PX,
    MAX_CLOUD_COVER,
    START_DATE,
)

STAC_CLIENT = pystac_client.Client.open(API_URL)


def get_epsg_from_coords(latitude: float, longitude: float) -> int:
    """
    Get the EPSG code for a specific coordinate

    Args:
        latitude (float):
            The latitude of the coordinate
        longitude (float):
            The longitude of the coordinate

    Returns:
        int:
            The EPSG code for the coordinate
    """
    crs_info = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(
            west_lon_degree=longitude,
            south_lat_degree=latitude,
            east_lon_degree=longitude,
            north_lat_degree=latitude,
        ),
    )
    return int(crs_info[0].code)


def create_aoi_for_plants(campd_facilities_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Create a square area of interest (AOI) for each plant in the CAMPD facilities data.
    This will later be used to query for satellite imagery.

    Args:
        campd_facilities_gdf (gpd.GeoDataFrame):
            The CAMPD facilities data frame

    Returns:
        gpd.GeoDataFrame:
            A data frame containing the AOIs for each plant
    """
    facility_dfs = list()
    for _, facility_df in tqdm(
        campd_facilities_gdf.groupby("facility_id"),
        total=campd_facilities_gdf.facility_id.nunique(),
        desc="Creating AOIs for plants",
    ):
        # identify what is the local CRS for the current facility,
        # based on its latitude and longitude
        epsg = get_epsg_from_coords(
            facility_df.latitude.mean(), facility_df.longitude.mean()
        )
        # convert to the local CRS, based on the coordinates
        facility_df = facility_df.to_crs(epsg=epsg)
        # buffer the geometry into a square that is ~3.2km on each side
        facility_df.geometry = facility_df.geometry.buffer(
            AOI_SIZE_METERS / 2, cap_style=3
        )
        # convert back to the global CRS
        facility_df = facility_df.to_crs(epsg=GLOBAL_EPSG)
        facility_dfs.append(facility_df)
    return gpd.GeoDataFrame(pd.concat(facility_dfs, ignore_index=True))


def get_aws_cog_links_from_geom(
    geometry: BaseGeometry,
    collection: str = COLLECTION,
    start_date: Optional[datetime] = START_DATE,
    end_date: Optional[datetime] = END_DATE,
    max_cloud_cover: Optional[int] = MAX_CLOUD_COVER,
    sort_by: str = "updated",
    max_items: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Retrieve links from AWS' Sentinel 2 L2A STAC

    Args:
        geometry (BaseGeometry):
            The geometry to query for images that
            contain it in STAC
        collection (str):
            The STAC collection to query
        start_date (Optional[datetime]):
            Optional start date to filter images on
        end_date (Optional[datetime]):
            Optional end date to filter images on
        max_cloud_cover (Optional[int]):
            Optional maximum cloud cover to filter
            images that are too cloudy. Expressed
            as a percentage, e.g. 1 = 1%
        sort_by (str):
            Which property to sort the results by,
            in descending order; needs to be a valid
            property in the STAC collection
        max_items (Optional[int]):
            Optional maximum number of items to
            return
        verbose (bool):
            Whether to print the progress of the
            query

    Returns:
        pd.DataFrame:
            A dataframe containing the ID of the tile and
            the links to its COGs and metadata
    """
    # get the bounding box from the geometry
    bbox = geometry.bounds
    # specify the cloud filter
    if max_cloud_cover == 0:
        cloud_filter = "eo:cloud_cover=0"
    elif max_cloud_cover is not None:
        cloud_filter = f"eo:cloud_cover<={max_cloud_cover}"
    # query the STAC collection(s) in a specific bounding box and search criteria
    search = STAC_CLIENT.search(
        collections=[collection],
        bbox=bbox,
        datetime=f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}",
        query=[cloud_filter] if max_cloud_cover is not None else None,
    )
    if verbose:
        logger.info(f"Found {search.matched()} items matching the search criteria")
    items = search.get_all_items()
    if max_cloud_cover is not None and collection == "sentinel-s2-l2a-cogs":
        # some items had invalid cloud cover data and turned out very cloudy; only works for L2A
        items_valid_cloud_filter = [
            x for x in items if x.properties["sentinel:valid_cloud_cover"] == True
        ]
        if verbose:
            logger.info(
                f"Removed {len(items) - len(items_valid_cloud_filter)} items for invalid cloud filters"
            )
        items = items_valid_cloud_filter
    items = sorted(items, key=lambda x: x.properties[sort_by], reverse=True)
    if max_items is not None:
        items = items[:max_items]
    # create a dictionary that contains the tile ID and the links to the COGs and metadata
    output = dict(tile_id=[item.id for item in items])
    asset_keys = items[0].assets.keys()
    for key in asset_keys:
        output[key] = [item.assets[key].href for item in items]
    output["cloud_cover"] = [item.properties["eo:cloud_cover"] for item in items]
    output[sort_by] = [item.properties[sort_by] for item in items]
    output["ts"] = [item.properties["datetime"] for item in items]
    output = pd.DataFrame(output)
    output["ts"] = pd.to_datetime(output["ts"])
    output.drop_duplicates(subset="ts", keep="first", inplace=True)
    output.sort_values("ts", inplace=True)
    return output


def get_image_metadata_for_plants(
    plant_aoi_gdf: gpd.GeoDataFrame,
    collection: str = COLLECTION,
    start_date: datetime = START_DATE,
    end_date: datetime = END_DATE,
    max_cloud_cover: int = MAX_CLOUD_COVER,
    sort_by: str = "updated",
) -> pd.DataFrame:
    """
    Get the metadata for the satellite images for each plant,
    based on the AOI defined for each plant (see create_aoi_for_plants)

    Args:
        plant_aoi_gdf (gpd.GeoDataFrame):
            The data frame containing the AOIs for each plant
        collection (str):
            The STAC collection to query
        start_date (Optional[datetime]):
            Start date to filter images on
        end_date (Optional[datetime]):
            End date to filter images on
        max_cloud_cover (Optional[int]):
            Maximum cloud cover to filter
            images that are too cloudy. Expressed
            as a percentage, e.g. 1 = 1%
        sort_by (str):
            Which property to sort the results by,
            in descending order; needs to be a valid
            property in the STAC collection

    Returns:
        pd.DataFrame:
            A dataframe containing the ID of the tile and
            the links to its COGs and metadata
    """
    image_metadata_dfs = list()
    for facility_id, geometry in tqdm(
        plant_aoi_gdf.groupby("facility_id").geometry.first().items(),
        total=plant_aoi_gdf.facility_id.nunique(),
        desc="Querying STAC API",
    ):
        stac_results_df = get_aws_cog_links_from_geom(
            geometry=geometry,
            collection=collection,
            start_date=start_date,
            end_date=end_date,
            max_cloud_cover=max_cloud_cover,
            sort_by=sort_by,
            verbose=False,
        )
        stac_results_df["facility_id"] = facility_id
        image_metadata_dfs.append(stac_results_df)
    return pd.concat(image_metadata_dfs, ignore_index=True)


def pad_or_crop_to_size(image: np.ndarray, size: int = IMAGE_SIZE_PX) -> np.ndarray:
    """
    Pad or crop an image to a specific size

    Args:
        image (np.ndarray):
            The image to pad or crop, with dimensions (C, H, W),
            where C is the number of channels, H is the height and
            W is the width
        size (int):
            The size to pad or crop to

    Returns:
        np.ndarray:
            The padded or cropped image
    """
    if image.shape[1] > size:
        # crop the image
        image = image[:, :size, :size]
    elif image.shape[1] < size:
        # pad the image
        image = np.pad(
            image,
            ((0, 0), (0, size - image.shape[1]), (0, size - image.shape[2])),
        )
    return image


def get_image_from_cog(
    cog_url: str, geometry: BaseGeometry, size: int = IMAGE_SIZE_PX
) -> np.ndarray:
    """
    Get the image from a COG, clipped to the geometry

    Args:
        cog_url (str):
            The URL to the COG
        geometry (BaseGeometry):
            The geometry to clip the image to
        size (int):
            The size to pad or crop to

    Returns:
        np.ndarray:
            The clipped image
    """
    # load only the bbox of the image
    with rio.open(cog_url) as src:
        # get the bbox converted to the right coordinate reference system (crs);
        # doing all of this because geopandas has the convenient to_crs function
        crs_bbox = (
            gpd.GeoDataFrame(geometry=[geometry], crs=GLOBAL_EPSG)
            .to_crs(src.crs)
            .total_bounds
        )
        # define window in RasterIO
        window = rio.windows.from_bounds(*crs_bbox, transform=src.transform)
        # actual HTTP range request
        image = src.read(window=window)
    # make sure that the image has the shape that we want
    image = pad_or_crop_to_size(image, size=size)
    return image
