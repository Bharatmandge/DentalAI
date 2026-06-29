import os
import requests
import nibabel as nib
import numpy as np

os.makedirs("data/raw", exist_ok=True)

# Working raw GitHub NIfTI file
url = "https://raw.githubusercontent.com/neurolabusc/niivue-images/master/mni152.nii.gz"

image_path = "data/raw/real_scan.nii.gz"
label_path = "data/raw/real_label.nii.gz"

print("Downloading NIfTI scan...")

headers = {
    "User-Agent": "Mozilla/5.0"
}

response = requests.get(url, headers=headers, stream=True, timeout=60)

if response.status_code != 200:
    raise Exception(f"Download failed! HTTP {response.status_code}")

with open(image_path, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)

print("Download Complete!")

print("Loading image...")
img = nib.load(image_path)
data = img.get_fdata()

print("Generating segmentation mask...")

threshold = np.mean(data) + np.std(data)

mask = (data > threshold).astype(np.float32)

mask_img = nib.Nifti1Image(mask, img.affine)

nib.save(mask_img, label_path)

print("\nSUCCESS!")
print(f"Image : {image_path}")
print(f"Mask  : {label_path}")