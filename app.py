from flask import Flask, request, render_template
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import numpy as np

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
if os.path.exists(model_path):
    model = MildewCNN()  
    model.load_state_dict(torch.load(model_path, weights_only=True))  
    model.eval()  
    print("Model loaded successfully")
else:
    raise FileNotFoundError(f"Model file {model_path} not found")


transform = transforms.Compose([
    transforms.ToTensor(),  
])


def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    return "Healthy" if predicted.item() == 0 else "Powdery Mildew"

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            file_path = os.path.join('uploads', file.filename)
            os.makedirs('uploads', exist_ok=True)
            file.save(file_path)
            prediction = predict_image(file_path)
            return f"Prediction: {prediction}"
    return '''
    <h1>Upload Cherry Leaf Image</h1>
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="file">
        <input type="submit">
    </form>
    '''

if __name__ == '__main__':
    app.run(debug=True, port=5001)