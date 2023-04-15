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
    assigned_facilities = set()
    for facility_id, facility_gdf in campd_facilities_gdf.groupby("facility_id"):
        if facility_id in assigned_facilities:
            continue
        # assign a data set to the facility
        data_set = np.random.choice(
            ["train", "val"], p=[train_val_ratio, 1 - train_val_ratio]
        )
        campd_facilities_gdf.loc[
            campd_facilities_gdf.facility_id == facility_id, "data_set"
        ] = data_set
        assigned_facilities.add(facility_id)
        # apply the same data set to intersecting facilities
        other_facilities_gdf = campd_facilities_gdf.loc[
            campd_facilities_gdf.facility_id != facility_id, ["facility_id", "geometry"]
        ]
        other_facilities_gdf.rename(
            columns={"facility_id": "intersecting_facility_id"}, inplace=True
        )
        intersecting_facilities_gdf = gpd.sjoin(
            facility_gdf,
            other_facilities_gdf,
            how="inner",
            predicate="intersects",
        )
        if intersecting_facilities_gdf.empty:
            continue
        else:
            for intersecting_facility_id in intersecting_facilities_gdf[
                "intersecting_facility_id"
            ].unique():
                campd_facilities_gdf.loc[
                    campd_facilities_gdf.facility_id == intersecting_facility_id,
                    "data_set",
                ] = data_set
                assigned_facilities.add(intersecting_facility_id)
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
    if row.ts.year == test_year:
        data_set = "test"
    else:
        data_set = data_set_mapper[row.facility_id]
    return data_set
