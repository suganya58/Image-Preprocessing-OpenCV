import os, zipfile, cv2, numpy as np, matplotlib.pyplot as plt
from kaggle.api.kaggle_api_extended import KaggleApi
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --------------------------------------------
# STEP 1: Kaggle dataset download & extraction
# --------------------------------------------
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)

api = KaggleApi()
api.authenticate()

api.dataset_download_files('sthabile/noisy-and-rotated-scanned-documents', path='.', unzip=False)

zip_path = "noisy-and-rotated-scanned-documents.zip"
extract_dir = "dataset"
os.makedirs(extract_dir, exist_ok=True)

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_dir)
print(" Dataset extracted successfully!")

# --------------------------------------------
# STEP 2: Collect all image files
# --------------------------------------------
image_files = []
for root, dirs, files in os.walk(extract_dir):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_files.append(os.path.join(root, f))

if not image_files:
    raise FileNotFoundError("No images found in dataset!")
else:
    print(f" Found {len(image_files)} images.")

# --------------------------------------------
# STEP 3: Image preprocessing function
# --------------------------------------------
output_dir = "cleaned_images"
os.makedirs(output_dir, exist_ok=True)

def clean_image_tuned(image_path, output_dir):
    img = cv2.imread(image_path)
    if img is None:
        print(" Skipping:", image_path)
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=50)

    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
    enhanced = clahe.apply(sharpened)

    _, final = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    filename = os.path.basename(image_path)
    cleaned_path = os.path.join(output_dir, filename)
    cv2.imwrite(cleaned_path, final)

    return cleaned_path

# --------------------------------------------
# STEP 4: Process all images
# --------------------------------------------
cleaned_paths = []
for img_path in image_files:
    result_path = clean_image_tuned(img_path, output_dir)
    if result_path:
        cleaned_paths.append(result_path)

print(f"Cleaned {len(cleaned_paths)} images. Saved in: {output_dir}")

# --------------------------------------------
# STEP 5: Integrate PyTesseract OCR
# --------------------------------------------
# (If you’re using Windows, set the tesseract.exe path)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def perform_ocr(image_path, output_text_dir="ocr_texts", visualize=True):
    os.makedirs(output_text_dir, exist_ok=True)
    img = cv2.imread(image_path)

    # OCR text extraction
    extracted_text = pytesseract.image_to_string(img)
    filename = os.path.splitext(os.path.basename(image_path))[0]

    # Save text to file
    text_path = os.path.join(output_text_dir, f"{filename}.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(extracted_text)

    print(f" OCR text saved for {filename}")

    # Optional visualization of detected boxes
    if visualize:
        boxes = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        for i in range(len(boxes['text'])):
            if int(boxes['conf'][i]) > 60:  # confidence threshold
                x, y, w, h = boxes['left'][i], boxes['top'][i], boxes['width'][i], boxes['height'][i]
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"OCR Visualization - {filename}")
        plt.axis('off')
        plt.show()

# --------------------------------------------
# STEP 6: Run OCR on one cleaned sample
# --------------------------------------------
sample_image = cleaned_paths[0]
perform_ocr(sample_image)
