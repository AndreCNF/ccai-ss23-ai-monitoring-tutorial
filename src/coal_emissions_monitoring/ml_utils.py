from typing import Dict, Tuple
import geopandas as gpd
import numpy as np
import pandas as pd
import torch

from coal_emissions_monitoring.data_cleaning import load_clean_campd_facilities_gdf
from coal_emissions_monitoring.constants import TEST_YEAR, TRAIN_VAL_RATIO


def get_facility_set_mapper(
    gdf: gpd.GeoDataFrame, train_val_ratio: float = TRAIN_VAL_RATIO
) -> Dict[int, str]:
    """
    Get a mapper from facility ID to a set of train or validation.

    Args:
        gdf (gpd.GeoDataFrame):
            The gdf containing the facility IDs
        train_val_ratio (float):
            The ratio of training to validation data

    Returns:
        Dict[int, str]:
            A mapper from facility ID to a set of train or validation
    """
    assigned_facilities = set()
    for facility_id, facility_gdf in gdf.groupby("facility_id"):
        if facility_id in assigned_facilities:
            continue
        # assign a data set to the facility
        data_set = np.random.choice(
            ["train", "val"], p=[train_val_ratio, 1 - train_val_ratio]
        )
        gdf.loc[gdf.facility_id == facility_id, "data_set"] = data_set
        assigned_facilities.add(facility_id)
        # apply the same data set to intersecting facilities
        other_facilities_gdf = gdf.loc[
            gdf.facility_id != facility_id, ["facility_id", "geometry"]
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
                gdf.loc[
                    gdf.facility_id == intersecting_facility_id,
                    "data_set",
                ] = data_set
                assigned_facilities.add(intersecting_facility_id)
    # create a mapper from facility ID to a set of train or validation
    return gdf.groupby("facility_id").data_set.first().to_dict()


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


def emissions_to_category(
    emissions: float, quantiles: Dict[float, float], rescale: bool = False
) -> int:
    """
    Convert emissions to a category based on quantiles. The quantiles are
    calculated from the training data. Here's how the categories are defined:
    - 0: no emissions
    - 1: low emissions
    - 2: medium emissions
    - 3: high emissions
    - 4: very high emissions

    Args:
        emissions (float): emissions value
        quantiles (Dict[float, float]): quantiles to use for categorization
        rescale (bool): whether to rescale emissions to the original range,
            using the 99th quantile as the maximum value

    Returns:
        int: category
    """
    if rescale:
        emissions = emissions * quantiles[0.99]
    if emissions <= 0:
        return 0
    elif emissions <= quantiles[0.3]:
        return 1
    elif emissions > quantiles[0.3] and emissions <= quantiles[0.6]:
        return 2
    elif emissions > quantiles[0.6] and emissions <= quantiles[0.99]:
        return 3
    else:
        return 4


def preds_n_targets_to_categories(
    preds: torch.Tensor,
    targets: torch.Tensor,
    quantiles: Dict[float, float],
    rescale: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert emissions to a category based on quantiles. The quantiles are
    calculated from the training data. Here's how the categories are defined:
    - 0: no emissions
    - 1: low emissions
    - 2: medium emissions
    - 3: high emissions
    - 4: very high emissions

    Args:
        preds (torch.Tensor): emissions predictions
        targets (torch.Tensor): emissions targets
        quantiles (Dict[float, float]): quantiles to use for categorization
        rescale (bool): whether to rescale emissions to the original range,
            using the 99th quantile as the maximum value

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: tuple of predictions and targets
    """
    preds_cat = torch.tensor(
        [
            emissions_to_category(y_pred_i, quantiles, rescale=rescale)
            for y_pred_i in preds
        ]
    ).to(preds.device)
    targets_cat = torch.tensor(
        [emissions_to_category(y_i, quantiles, rescale=rescale) for y_i in targets]
    ).to(targets.device)
    return preds_cat, targets_cat
