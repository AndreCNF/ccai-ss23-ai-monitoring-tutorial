from datetime import datetime

GLOBAL_EPSG = 4326
API_URL = "https://earth-search.aws.element84.com/v0"
COLLECTION = "sentinel-s2-l2a-cogs"  # Sentinel-2, Level 2A, COGs
AOI_SIZE_METERS = 3200
IMAGE_SIZE_PX = 352
CROP_SIZE_PX = 256
START_DATE = datetime(year=2019, month=1, day=1)
END_DATE = datetime(year=2023, month=3, day=31)
MAX_DARK_FRAC = 0.5
MAX_CLOUD_COVER_PRCT = 25
TRAIN_VAL_RATIO = 0.8
TEST_YEAR = 2023
BATCH_SIZE = 32
MAIN_COLUMNS = [
    "facility_id",
    "facility_name",
    "latitude",
    "longitude",
    "ts",
    "co2_mass_short_tons",
    "cloud_cover",
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
EMISSIONS_TARGET = "co2_mass_short_tons"
RANDOM_TRANSFORM_PROB = 0.5
