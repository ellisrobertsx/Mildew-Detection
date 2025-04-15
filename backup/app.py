from flask import Flask, request, render_template, send_from_directory, abort, redirect, Response
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from google.cloud import storage
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import json
import csv
from io import StringIO

logging.basicConfig(level=logging.DEBUG)

class MildewCNN(nn.Module):
    def __init__(self):
        super(MildewCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 64 * 64, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

app = Flask(__name__)

model_path = 'mildew_cnn_model_trained.pth'
bucket_name = 'mildew-detection-uploads-2025'
gcs_model_path = 'mildew_cnn_model_trained.pth'

# Check if running on Heroku (PORT env var is set by Heroku)
IS_HEROKU = 'PORT' in os.environ

if IS_HEROKU:
    if 'GOOGLE_APPLICATION_CREDENTIALS_JSON' in os.environ:
        creds_json = os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON']
        try:
            with open('mildew-app-key.json', 'w') as f:
                f.write(creds_json)
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'mildew-app-key.json'
            logging.info("Credentials loaded from config var")
        except Exception as e:
            logging.error(f"Failed to write credentials: {str(e)}")
            raise
    else:
        logging.error("No GOOGLE_APPLICATION_CREDENTIALS_JSON in environment")
        raise FileNotFoundError("GCS credentials not found in environment")

def download_model_from_gcs():
    if not os.path.exists(model_path):
        print(f"Downloading model from GCS: gs://{bucket_name}/{gcs_model_path}")
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(gcs_model_path)
        blob.download_to_filename(model_path)
        print("Model downloaded successfully")
    else:
        print("Model already exists locally")

def upload_to_gcs(file, filename):
    logging.info(f"Uploading {filename} to GCS bucket: {bucket_name}")
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f"uploads/{filename}")
        blob.upload_from_file(file, rewind=True, content_type=file.content_type)
        logging.info(f"Uploaded {filename} successfully")
        return f"https://storage.googleapis.com/{bucket_name}/uploads/{filename}"
    except Exception as e:
        logging.error(f"GCS upload failed for {filename}: {str(e)}")
        raise

try:
    if IS_HEROKU:
        download_model_from_gcs()
    else:
        print("Running locally, skipping GCS model download. Ensure mildew_cnn_model_trained.pth exists.")
    model = MildewCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Model loaded successfully")
except Exception as e:
    raise FileNotFoundError(f"Failed to load model: {str(e)}")

transform = transforms.Compose([transforms.ToTensor()])

def predict_image(image_path):
    print(f"Processing image: {image_path}")
    if not os.path.isfile(image_path) or not image_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        raise ValueError("Invalid image file")
    try:
        image = Image.open(image_path).convert('RGB')
        image = image.resize((256, 256))
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(image)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            prediction = "Healthy" if predicted.item() == 0 else "Powdery Mildew"
            probability = confidence.item() * 100  # Convert to percentage
        return prediction, probability
    except Exception as e:
        return f"Error predicting: {str(e)}", 0

def plot_training_metrics():
    # Real training history from 03_model_training.ipynb (trimmed to 5 epochs)
    history = {
        'train_loss': [0.3083, 0.0614, 0.0152, 0.0042, 0.0031],
        'val_loss': [0.0264, 0.0195, 0.0066, 0.0130, 0.0165],
        'train_acc': [0.8526, 0.9769, 0.9959, 0.9993, 1.0],
        'val_acc': [0.9905, 0.9905, 0.9937, 0.9937, 0.9937]
    }

    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    loss_plot = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    acc_plot = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    # Real performance metrics from test set evaluation
    metrics = {
        'precision_healthy': 0.9751552795031055,
        'recall_healthy': 0.9936708860759493,
        'f1_healthy': 0.9843260188087775,
        'precision_mildew': 0.9935483870967742,
        'recall_mildew': 0.9746835443037974,
        'f1_mildew': 0.9840255591054313,
        'class_accuracy': {
            'Healthy': 0.9936708860759493,
            'Powdery Mildew': 0.9746835443037974
        }
    }

    # Real confusion matrix from test set
    confusion_matrix = np.array([[157, 1], [4, 154]])

    # Plot confusion matrix heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Healthy', 'Powdery Mildew'], 
                yticklabels=['Healthy', 'Powdery Mildew'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    cm_plot = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    # Plot class-wise accuracy bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(metrics['class_accuracy'].keys(), metrics['class_accuracy'].values(), color=['#4CAF50', '#FF5733'])
    plt.title('Class-wise Accuracy')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, v in enumerate(metrics['class_accuracy'].values()):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    class_acc_plot = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    # Real precision-recall curves from test set
    precision_healthy = [0.5, 0.5015873, 0.50318471, 0.50479233, 0.50641026]
    recall_healthy = [1.0, 1.0, 1.0, 1.0, 1.0]
    precision_mildew = [0.5, 0.57664234, 0.59398496, 0.60536398, 0.62450593]
    recall_mildew = [1.0, 1.0, 1.0, 1.0, 1.0]

    plt.figure(figsize=(10, 5))
    plt.plot(recall_healthy, precision_healthy, label='Healthy', color='#4CAF50')
    plt.plot(recall_mildew, precision_mildew, label='Powdery Mildew', color='#FF5733')
    plt.title('Precision-Recall Curves')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    pr_plot = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    # Real model comparison
    model_comparison = {
        'MildewCNN': 0.9937
    }

    plt.figure(figsize=(10, 5))
    plt.bar(model_comparison.keys(), model_comparison.values(), color=['#4CAF50'])
    plt.title('Model Comparison: Test Accuracy')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, v in enumerate(model_comparison.values()):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    plt.xticks(rotation=45)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    model_comp_plot = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    return loss_plot, acc_plot, cm_plot, class_acc_plot, pr_plot, model_comp_plot, metrics

@app.route('/Uploads/<filename>')
def uploaded_file(filename):
    try:
        return send_from_directory('Uploads', filename, as_attachment=False)
    except FileNotFoundError:
        abort(404)
    except Exception as e:
        abort(500, description=str(e))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    predictions = None
    predictions_json = None
    if request.method == 'POST':
        logging.info("Received POST request")
        if 'files' not in request.files:
            logging.error("No file part in request")
            return "No file part"
        files = request.files.getlist('files')
        if not files or all(file.filename == '' for file in files):
            logging.error("No selected files")
            return "No selected files"
        predictions = []
        for file in files:
            if file:
                file_path = os.path.join('Uploads', file.filename)
                os.makedirs('Uploads', exist_ok=True)
                file.save(file_path)
                logging.info(f"Saved file to {file_path}")
                prediction, probability = predict_image(file_path)
                if "Error" in prediction:
                    logging.error(f"Prediction failed for {file.filename}: {prediction}")
                    continue
                if IS_HEROKU:
                    image_url = upload_to_gcs(file, file.filename)
                else:
                    image_url = f"/Uploads/{os.path.basename(file_path)}"
                    logging.info("Running locally, using local file path for image preview")
                predictions.append({
                    'filename': file.filename,
                    'prediction': prediction,
                    'probability': f"{probability:.2f}",
                    'image_url': image_url
                })
        if predictions:
            predictions_json = json.dumps(predictions)
        return render_template('predict.html', predictions=predictions, predictions_json=predictions_json)
    return render_template('predict.html', predictions=predictions, predictions_json=predictions_json)

@app.route('/download_predictions', methods=['POST'])
def download_predictions():
    predictions_json = request.form.get('predictions')
    predictions = json.loads(predictions_json) if predictions_json else []
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Image Name', 'Prediction', 'Probability (%)'])
    for pred in predictions:
        writer.writerow([pred['filename'], pred['prediction'], pred['probability']])
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={"Content-Disposition": "attachment;filename=predictions.csv"}
    )

@app.route('/performance')
def performance():
    loss_plot, acc_plot, cm_plot, class_acc_plot, pr_plot, model_comp_plot, metrics = plot_training_metrics()
    return render_template('performance.html', loss_plot=loss_plot, acc_plot=acc_plot, cm_plot=cm_plot, 
                          class_acc_plot=class_acc_plot, pr_plot=pr_plot, model_comp_plot=model_comp_plot, metrics=metrics)

@app.route('/dataset')
def dataset():
    return render_template('dataset.html')

@app.route('/visual_study')
def visual_study():
    return render_template('visual_study.html')

@app.route('/hypothesis')
def hypothesis():
    return render_template('hypothesis.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)