import os
import shutil
import zipfile
# %%
# zips the image data from directory into a zip file (zip_data.zip) to upload into google colab
#shutil.make_archive('zip_data', 'zip', 'COVID-img-data')
shutil.make_archive('zip_data', 'zip', 'data')
# %%
# extracts zip data into directory
path_to_zip_file = 'zip_data.zip'
#directory_to_extract_to = 'COVID-img-data'
directory_to_extract_to = 'data'
with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
    zip_ref.extractall(directory_to_extract_to)