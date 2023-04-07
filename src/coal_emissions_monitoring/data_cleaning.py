from pathlib import Path
from typing import Callable, Optional, Union

import pandas as pd
import geopandas as gpd

GLOBAL_CRS = "EPSG:4326"


def clean_column_names(
    df: Union[pd.DataFrame, gpd.GeoDataFrame]
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Clean column names in a data frame.

    Args:
        df (Union[pd.DataFrame, gpd.GeoDataFrame]):
            Data frame to clean

    Returns:
        df (Union[pd.DataFrame, gpd.GeoDataFrame]):
            Cleaned data frame
    """
    df.columns = (
        df.columns.str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "")
        .str.replace(")", "")
        .str.replace("/", "_")
        .str.replace("-", "_")
        .str.replace(",", "_")
    )
    return df


def load_clean_data_gdf(
    data_path: Union[str, Path],
    load_func: Optional[Callable] = pd.read_csv,
    clean_func: Optional[Callable] = clean_column_names,
) -> gpd.GeoDataFrame:
    """
    Load and clean a data frame.

    Args:
        data_path (Union[str, Path]):
            Path to data
        load_func (Optional[Callable]):
            Function to load data
        clean_func (Optional[Callable]):
            Function to clean data

    Returns:
        gdf (gpd.GeoDataFrame):
            Cleaned data frame
    """
    df = load_func(data_path)
    df = clean_func(df)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(
            df["longitude"],
            df["latitude"],
        ),
        crs=GLOBAL_CRS,
    )
    return gdf


def load_raw_gcpt_data(gcpt_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load GCPT data in its raw excel format from GCS.

    Returns:
        df (pd.DataFrame):
            GCPT data frame
    """
    df = pd.read_excel(
        gcpt_path,
        sheet_name="Units",
    )
    return df


def clean_gcpt(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the GCPT data frame, setting better column names.

    Args:
        df (pd.DataFrame):
            GCPT data frame

    Returns:
        df (pd.DataFrame):
            Cleaned GCPT data frame
    """
    df = clean_column_names(df)
    df.rename(columns={"parentid": "parent_id"}, inplace=True)
    df.rename(columns={"trackerloc": "tracker_loc"}, inplace=True)
    return df


def load_clean_gcpt_gdf(gcpt_path: Union[str, Path]) -> gpd.GeoDataFrame:
    """
    Load and clean the GCPT data frame.

    Args:
        gcpt_path (Union[str, Path]):
            Path to GCPT data

    Returns:
        gdf (gpd.GeoDataFrame):
            Cleaned GCPT data frame
    """
    return load_clean_data_gdf(
        data_path=gcpt_path, load_func=load_raw_gcpt_data, clean_func=clean_gcpt
    )


def clean_campd_facilities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the CAMPD facilities data frame.

    Args:
        df (pd.DataFrame):
            CAMPD facilities data frame

    Returns:
        df (pd.DataFrame):
            Cleaned CAMPD facilities data frame
    """
    df = clean_column_names(df)
    # get the capacity
    df["capacity_mw"] = (
        df["associated_generators_&_nameplate_capacity_mwe"]
        .str.split(" ")
        .str[-1]
        .str.replace("(", "")
        .str.replace(")", "")
        .astype(float)
    )
    # filter to operating units
    df = df[(df.operating_status == "Operating") & (df.capacity_mw > 0)]
    # aggregate by facility
    df = df.groupby(["facility_id", "year"]).agg(
        {
            "capacity_mw": "sum",
            "facility_name": "first",
            "latitude": "mean",
            "longitude": "mean",
        }
    )
    # rearrange columns
    df = df.reset_index()[
        ["facility_id", "facility_name", "year", "capacity_mw", "latitude", "longitude"]
    ]
    return df


def load_clean_campd_facilities_gdf(campd_path: Union[str, Path]) -> gpd.GeoDataFrame:
    """
    Load and clean the CAMPD facilities data frame.

    Args:
        campd_path (Union[str, Path]):
            Path to CAMPD facilities data

    Returns:
        gdf (gpd.GeoDataFrame):
            Cleaned CAMPD facilities data frame
    """
    return load_clean_data_gdf(
        data_path=campd_path, load_func=pd.read_csv, clean_func=clean_campd_facilities
    )
