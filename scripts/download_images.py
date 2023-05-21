# %% [markdown]
# # Images download
# ---
#
# Download all images before training models.

# %% [markdown]
# ## Setup

# %% [markdown]
# ### Imports

# %%
import os
from tqdm.auto import tqdm

# %%
from coal_emissions_monitoring.constants import ALL_BANDS
from coal_emissions_monitoring.data_cleaning import get_final_dataset
from coal_emissions_monitoring.satellite_imagery import fetch_image_path_from_cog

# %% [markdown]
# ## Get final datase

# %%
df = get_final_dataset(
    image_metadata_path="/home/adminuser/ccai-ss23-ai-monitoring-tutorial/data/image_metadata.csv",
    campd_facilities_path="https://drive.google.com/file/d/1b-5BriZUiiv2r0wFLubccLQpd2xb5ysl/view?usp=share_link",
    campd_emissions_path="https://drive.google.com/file/d/1oxZXR7GDcSXwwVoIjp66iS179cFVA5dP/view?usp=share_link",
    cog_type="all",
)

# %% [markdown]
# ## Download images

# %% [markdown]
# ### TCI (True Color Image)

# %%
tqdm.pandas(desc="Downloading visual images")
df["local_image_path"] = df.progress_apply(
    lambda row: fetch_image_path_from_cog(
        cog_url=row.visual,
        geometry=row.geometry,
        images_dir="/home/adminuser/ccai-ss23-ai-monitoring-tutorial/data/images/visual/",
        download_missing_images=True,
    ),
    axis=1,
)

# %%
# compress all images into one file
os.system(
    "tar -czvf /home/adminuser/ccai-ss23-ai-monitoring-tutorial/data/images/visual_images.tar.gz /home/adminuser/ccai-ss23-ai-monitoring-tutorial/data/images/visual"
)

# %% [markdown]
# ### All bands

# %%
tqdm.pandas(desc="Downloading all bands images")
df["local_image_all_bands_path"] = df.progress_apply(
    lambda row: fetch_image_path_from_cog(
        cog_url=[row[band] for band in ALL_BANDS],
        geometry=row.geometry,
        cog_type="all",
        images_dir="/home/adminuser/ccai-ss23-ai-monitoring-tutorial/data/images/all_bands/",
        download_missing_images=True,
    ),
    axis=1,
)

# %%
# compress all images into one file
os.system(
    "!tar -czvf /home/adminuser/ccai-ss23-ai-monitoring-tutorial/data/images/all_bands_images.tar.gz /home/adminuser/ccai-ss23-ai-monitoring-tutorial/data/images/all_bands"
)

# %%
df.to_csv(
    "/home/adminuser/ccai-ss23-ai-monitoring-tutorial/data/final_dataset.csv",
    index=False,
)
