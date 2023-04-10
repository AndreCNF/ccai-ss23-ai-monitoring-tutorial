from pathlib import Path
import warnings
from typing import Callable, Optional, Union

import pandas as pd
import geopandas as gpd
import overpy

from coal_emissions_monitoring.constants import GLOBAL_EPSG

OSM_API = overpy.Overpass()

# suppress geopandas CRS warning as we don't need to worry too much about
# the precision of distances
warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS*")


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


def load_clean_data_df(
    data_path: Union[str, Path],
    load_func: Optional[Callable] = pd.read_csv,
    clean_func: Optional[Callable] = clean_column_names,
) -> pd.DataFrame:
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
        df (pd.DataFrame):
            Cleaned data frame
    """
    df = load_func(data_path)
    df = clean_func(df)
    return df


def load_clean_data_gdf(
    data_path: Union[str, Path],
    load_func: Optional[Callable] = pd.read_csv,
    clean_func: Optional[Callable] = clean_column_names,
) -> gpd.GeoDataFrame:
    """
    Load and clean a data frame, outputting it as a GeoDataFrame.

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
    df = load_clean_data_df(
        data_path=data_path, load_func=load_func, clean_func=clean_func
    )
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(
            df["longitude"],
            df["latitude"],
        ),
        crs=f"EPSG:{GLOBAL_EPSG}",
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
    # fix datetime column data type
    df.year = pd.to_datetime(df.year, format="%Y")
    return df


def load_clean_campd_facilities_gdf(
    campd_facilities_path: Union[str, Path]
) -> gpd.GeoDataFrame:
    """
    Load and clean the CAMPD facilities data frame.

    Args:
        campd_facilities_path (Union[str, Path]):
            Path to CAMPD facilities data

    Returns:
        gdf (gpd.GeoDataFrame):
            Cleaned CAMPD facilities data frame
    """
    return load_clean_data_gdf(
        data_path=campd_facilities_path,
        load_func=pd.read_csv,
        clean_func=clean_campd_facilities,
    )


def clean_campd_emissions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the CAMPD emissions data frame.

    Args:
        df (pd.DataFrame):
            CAMPD emissions data frame

    Returns:
        df (pd.DataFrame):
            Cleaned CAMPD emissions data frame
    """
    df = clean_column_names(df)
    # fix datetime column data type
    df.date = pd.to_datetime(df.date)
    # fill missing values (emissions seem to be ignored if their value is 0)
    df = df.fillna(0)
    return df


def load_clean_campd_emissions_df(
    campd_emissions_path: Union[str, Path]
) -> pd.DataFrame:
    """
    Load and clean the CAMPD emissions data frame.

    Args:
        campd_emissions_path (Union[str, Path]):
            Path to CAMPD emissions data

    Returns:
        df (pd.DataFrame):
            Cleaned CAMPD emissions data frame
    """
    return load_clean_data_df(
        data_path=campd_emissions_path,
        load_func=pd.read_csv,
        clean_func=clean_campd_emissions,
    )


def load_osm_data(
    country: str = "United States", tag: str = "man_made", value: str = "cooling_tower"
) -> gpd.GeoDataFrame:
    """
    Load OSM data.

    Args:
        country (str):
            Country to filter to
        tag (str):
            OSM tag to filter to
        value (str):
            OSM value to filter to

    Returns:
        gdf (gpd.GeoDataFrame):
            OSM cooling towers data frame
    """
    # load the data
    osm_results = OSM_API.query(
        query=f"""
        area[name="{country}"]->.searchArea;
        (
        node["{tag}"="{value}"](area.searchArea);
        way["{tag}"="{value}"](area.searchArea);
        relation["{tag}"="{value}"](area.searchArea);
        );
        out body;
        >;
        out skel qt;
        """
    )
    df = pd.DataFrame(
        [
            {
                "osm_id": element.id,
                "latitude": element.lat,
                "longitude": element.lon,
            }
            for element in osm_results.nodes
        ]
    )
    # convert to geodataframe
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326",
    )
    return gdf


def clean_image_metadata(df: pd.DataFrame, cog_type: str = "visual") -> pd.DataFrame:
    """
    Clean the image metadata data frame.

    Args:
        df (pd.DataFrame):
            Image metadata data frame
        cog_type (str):
            Type of COG to filter to

    Returns:
        df (pd.DataFrame):
            Cleaned image metadata data frame
    """
    df = clean_column_names(df)
    # fix datetime column data type
    df.ts = pd.to_datetime(df.ts)
    # filter to most relevant columns
    df.rename(columns={cog_type: "cog_url"}, inplace=True)
    df = df[["facility_id", "ts", "cloud_cover", "cog_url"]]
    return df


def load_clean_image_metadata_df(
    image_metadata_path: Union[str, Path], cog_type: str = "visual"
) -> pd.DataFrame:
    """
    Load and clean the image metadata data frame.

    Args:
        image_metadata_path (Union[str, Path]):
            Path to image metadata data
        cog_type (str):
            Type of COG to filter to

    Returns:
        df (pd.DataFrame):
            Cleaned image metadata data frame
    """
    return load_clean_data_df(
        data_path=image_metadata_path,
        load_func=pd.read_csv,
        clean_func=lambda df: clean_image_metadata(df, cog_type=cog_type),
    )


def join_image_metadata_and_emissions(
    image_metadata_df: pd.DataFrame, emissions_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Join the image metadata and emissions data frames.

    Args:
        image_metadata_df (pd.DataFrame):
            Image metadata data frame
        emissions_df (pd.DataFrame):
            Emissions data frame

    Returns:
        df (pd.DataFrame):
            Joined image metadata and emissions data frame
    """
    # remove the hour info from the date so as to join by day of the year
    image_metadata_df["date_without_time"] = image_metadata_df["ts"].dt.date
    emissions_df["date_without_time"] = emissions_df["date"].dt.date
    # merge the two data frames
    merged_df = pd.merge(
        left=emissions_df,
        right=image_metadata_df,
        how="inner",
        on=["facility_id", "date_without_time"],
    )
    # drop the date without time column
    merged_df.drop(columns=["date_without_time"], inplace=True)
    return merged_df
