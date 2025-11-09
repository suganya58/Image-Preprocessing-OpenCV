from flask import Flask, render_template, request, send_from_directory
import os
from werkzeug.utils import secure_filename
import cv2
import pytesseract

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('result.html', label="No file uploaded", confidence=None, filename=None)

    file = request.files['file']
    if file.filename == '':
        return render_template('result.html', label="No selected file", confidence=None, filename=None)

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Read and process the image
    image = cv2.imread(file_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)

    # Simple classification logic (you can improve this)
    text_length = len(text.strip())
    if text_length > 25:
        label = "✅ Authentic (Text Image)"
        confidence = min(100, text_length / 2)
    else:
        label = "❌ Non-Authentic (Non-Text Image)"
        confidence = 100 - min(100, text_length * 2)

    # Pass values to the result page
    return render_template(
        'result.html',
        label=label,
        confidence=round(confidence, 2),
        filename=filename
    )

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
