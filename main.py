import os, zipfile, cv2, numpy as np, matplotlib.pyplot as plt
from kaggle.api.kaggle_api_extended import KaggleApi

# Setup Kaggle API
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)

os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)

api = KaggleApi()
api.authenticate()

# Download dataset
api.dataset_download_files('sthabile/noisy-and-rotated-scanned-documents', path='.', unzip=False)

# Extract dataset
zip_path = "noisy-and-rotated-scanned-documents.zip"
extract_dir = "dataset"
os.makedirs(extract_dir, exist_ok=True)
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_dir)
print("Dataset extracted successfully!")
# Collect image files
image_files = []
for root, dirs, files in os.walk(extract_dir):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_files.append(os.path.join(root, f))

if not image_files:
    raise FileNotFoundError("No images found in dataset!")
else:
   print(f"Found {len(image_files)} images.")

# Output directory
output_dir = "cleaned_images"
os.makedirs(output_dir, exist_ok=True)

# Image cleaning function
def clean_image_tuned(image_path, output_dir):
    img = cv2.imread(image_path)
    if img is None:
        print(" Skipping:", image_path)
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=50)

    kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
    enhanced = clahe.apply(sharpened)

    _, final = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    filename = os.path.basename(image_path)
    cv2.imwrite(os.path.join(output_dir, filename), final)

# Process all images
for img_path in image_files:
    clean_image_tuned(img_path, output_dir)

print(f" Cleaned {len(image_files)} images. Saved in: {output_dir}")

# Display sample image
sample_path = image_files[0]
orig = cv2.imread(sample_path)
cleaned = cv2.imread(os.path.join(output_dir, os.path.basename(sample_path)), cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Original Noisy Image")
plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.subplot(1,2,2)
plt.title("Cleaned Image")
plt.imshow(cleaned, cmap='gray')
plt.axis('off')
plt.show()