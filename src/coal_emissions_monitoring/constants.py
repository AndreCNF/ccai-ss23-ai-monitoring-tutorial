from datetime import datetime

GLOBAL_EPSG = 4326
API_URL = "https://earth-search.aws.element84.com/v0"
COLLECTION = "sentinel-s2-l2a-cogs"  # Sentinel-2, Level 2A, COGs
AOI_SIZE_METERS = 3200
IMAGE_SIZE_PX = 320
CROP_SIZE_PX = 256
START_DATE = datetime(year=2019, month=1, day=1)
END_DATE = datetime(year=2023, month=3, day=31)
MAX_CLOUD_COVER = 1
TRAIN_VAL_RATIO = 0.8
TEST_YEAR = 2023
BATCH_SIZE = 32
FINAL_COLUMNS = [
    "facility_id",
    "facility_name",
    "latitude",
    "longitude",
    "ts",
    "co2_mass_short_tons",
    "cloud_cover",
    "cog_url",
    "geometry",
]
EMISSIONS_TARGET = "co2_mass_short_tons"
