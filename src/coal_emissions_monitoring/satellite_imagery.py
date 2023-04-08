from typing import Optional
import pandas as pd
import pystac_client
from loguru import logger
from shapely.geometry.base import BaseGeometry
from datetime import datetime


def get_aws_cog_links_from_geom(
    stac_client: pystac_client.Client,
    geometry: BaseGeometry,
    collection: str = "sentinel-s2-l2a-cogs",
    start_date: Optional[datetime] = datetime(2022, 1, 1),
    end_date: Optional[datetime] = datetime(2022, 12, 31),
    max_cloud_cover: Optional[int] = None,
    sort_by: str = "updated",
    max_items: Optional[int] = None,
) -> pd.DataFrame:
    """
    Retrieve links from AWS' Sentinel 2 L2A STAC

    Args:
        client (pystac_client.Client):
            The `pystac_client` that queries STAC
        geometry (BaseGeometry):
            The geometry to query for images that
            contain it in STAC
        collection (str):
            The STAC collection to query
            For Sentinel 2 L2A, this is
            "sentinel-s2-l2a-cogs";
            For Sentinel 2 L1C, this is
            "sentinel-s2-l1c"
        start_date (Optional[datetime]):
            Optional start date to filter images on
        end_date (Optional[datetime]):
            Optional end date to filter images on
        max_cloud_cover (Optional[int]):
            Optional maximum cloud cover to filter
            images that are too cloudy
        sort_by (str):
            Which property to sort the results by,
            in descending order; needs to be a valid
            property in the STAC collection
        max_items (Optional[int]):
            Optional maximum number of items to
            return

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
    search = stac_client.search(
        collections=[collection],
        bbox=bbox,
        datetime=f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}",
        query=[cloud_filter] if max_cloud_cover is not None else None,
    )
    logger.info(f"Found {search.matched()} items matching the search criteria")
    items = search.get_all_items()
    if max_cloud_cover is not None and collection == "sentinel-s2-l2a-cogs":
        # some items had invalid cloud cover data and turned out very cloudy; only works for L2A
        items_valid_cloud_filter = [
            x for x in items if x.properties["sentinel:valid_cloud_cover"] == True
        ]
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
    if output_dataframe:
        output = pd.DataFrame(output)
    return output
