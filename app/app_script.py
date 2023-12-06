#got the requirements with this command:     pip freeze > requirements.txt


#     conda create -n envConda_docker python=3.11
#     conda activate envConda_docker
#     pip install -r requirements.txt

import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# Load the pre-trained model
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(2048, num_classes)  # Replace num_classes with the actual number of classes in your model
model.load_state_dict(torch.load('../train/model.pth'))
model.eval()

# Preprocess input image
def preprocess_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

# Perform inference
def predict_image_class(image_path):
    input_batch = preprocess_image(image_path)
    with torch.no_grad():
        output = model(input_batch)
    # Process the output as needed
    _, predicted = torch.max(output.data, 1)
    return predicted


# Usage
image_path = 'predict1.webp'
predicted_class = predict_image_class(image_path)
print(f"Predicted class for first image: {predicted_class}")

image_path = 'predict2.avif'
predicted_class = predict_image_class(image_path)
print(f"Predicted class for second image: {predicted_class}")

image_path = 'predict3.jpg'
predicted_class = predict_image_class(image_path)
print(f"Predicted class for third image: {predicted_class}")
