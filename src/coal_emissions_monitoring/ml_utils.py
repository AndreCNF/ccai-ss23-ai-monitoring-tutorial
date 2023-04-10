from typing import Dict
import geopandas as gpd
import numpy as np
import pandas as pd

from coal_emissions_monitoring.data_cleaning import load_clean_campd_facilities_gdf
from coal_emissions_monitoring.constants import TEST_YEAR, TRAIN_VAL_RATIO


def get_facility_set_mapper(
    campd_facilities_path: str, train_val_ratio: float = TRAIN_VAL_RATIO
) -> Dict[int, str]:
    """
    Get a mapper from facility ID to a set of train or validation.

    Args:
        campd_facilities_path (str):
            The path to the CAMPD facilities GeoDataFrame
        train_val_ratio (float):
            The ratio of training to validation data

    Returns:
        Dict[int, str]:
            A mapper from facility ID to a set of train or validation
    """
    campd_facilities_gdf = load_clean_campd_facilities_gdf(campd_facilities_path)
    # find distance to the nearest facility
    for facility_id, facility_gdf in campd_facilities_gdf.groupby("facility_id"):
        other_facilities_gdf = campd_facilities_gdf.loc[
            campd_facilities_gdf.facility_id != facility_id, ["facility_id", "geometry"]
        ]
        other_facilities_gdf.rename(
            columns={"facility_id": "nearest_facility_id"}, inplace=True
        )
        nearest_facility_gdf = gpd.sjoin_nearest(
            facility_gdf,
            other_facilities_gdf,
            distance_col="dist_to_nearest_facility",
            max_distance=0.1,
        ).head(1)
        if nearest_facility_gdf.empty:
            continue
        campd_facilities_gdf.loc[
            campd_facilities_gdf.facility_id == facility_id, "dist_to_nearest_facility"
        ] = nearest_facility_gdf.dist_to_nearest_facility
        campd_facilities_gdf.loc[
            campd_facilities_gdf.facility_id == facility_id, "nearest_facility_id"
        ] = nearest_facility_gdf.nearest_facility_id
    # assign a data set to each facility
    for facility_id, facility_df in campd_facilities_gdf.groupby("facility_id"):
        data_set = np.random.choice(
            ["train", "val"], p=[train_val_ratio, 1 - train_val_ratio]
        )
        campd_facilities_gdf.loc[
            campd_facilities_gdf.facility_id == facility_id, "data_set"
        ] = data_set
        nearest_facility_id = facility_df.nearest_facility_id.values[0]
        if not np.isnan(nearest_facility_id):
            # apply the same data set to the nearest facility
            campd_facilities_gdf.loc[
                campd_facilities_gdf.facility_id == nearest_facility_id,
                "data_set",
            ] = data_set
    # create a mapper from facility ID to a set of train or validation
    return campd_facilities_gdf.groupby("facility_id").data_set.first().to_dict()


def split_data_in_sets(
    row: pd.DataFrame, data_set_mapper: Dict[int, str], test_year: int = TEST_YEAR
) -> str:
    """
    Split the data in sets. This function is meant to be used with pandas.DataFrame.apply.

    Args:
        row (pd.DataFrame):
            The row of the DataFrame
        data_set_mapper (Dict[int, str]):
            A mapper from facility ID to a set of train or validation
        test_year (int):
            The year to use for testing

    Returns:
        str:
            The data set
    """
    if row.ts.dt.year == test_year:
        data_set = "test"
    else:
        data_set = data_set_mapper[row.facility_id]
    return data_set
