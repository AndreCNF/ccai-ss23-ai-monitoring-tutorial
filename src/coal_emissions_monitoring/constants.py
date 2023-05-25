from datetime import datetime

GLOBAL_EPSG = 4326
API_URL = "https://earth-search.aws.element84.com/v0"
COLLECTION = "sentinel-s2-l2a-cogs"  # Sentinel-2, Level 2A, COGs
AOI_SIZE_METERS = 640
IMAGE_SIZE_PX = 64
CROP_SIZE_PX = 52
START_DATE = datetime(year=2016, month=1, day=1)
END_DATE = datetime(year=2019, month=12, day=31)
MAX_DARK_FRAC = 0.5
MAX_BRIGHT_MEAN = 250
MAX_CLOUD_COVER_PRCT = 50
TRAIN_VAL_RATIO = 0.8
TEST_YEAR = 2020
BATCH_SIZE = 32
MAIN_COLUMNS = [
    "facility_id",
    "latitude",
    "longitude",
    "ts",
    "is_powered_on",
    "cloud_cover",
    "cog_url",
    "geometry",
]
ALL_BANDS = [
    "b01",
    "b02",
    "b03",
    "b04",
    "b05",
    "b06",
    "b07",
    "b08",
    "b8a",
    "b09",
    "b11",
    "b12",
]
EMISSIONS_TARGET = "is_powered_on"
EMISSIONS_CATEGORIES = {
    0: "no_emissions",
    1: "low",
    2: "medium",
    3: "high",
    4: "very_high",
}
RANDOM_TRANSFORM_PROB = 0.5
POSITIVE_THRESHOLD = 0.5
