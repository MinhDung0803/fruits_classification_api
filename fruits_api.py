import torch
import os
import io
from dotenv import load_dotenv
from models import ResNet_model
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image
import torch.backends.cudnn as cudnn
from flask import Flask, jsonify
from flask_cors import CORS

env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv(dotenv_path=env_path)

# config parameters
gpu_ids = int(os.getenv("GPU_ID"))
nclasses = int(os.getenv("CLASSES"))  # Number of classes in model

# model
model = None

# flask app
app = Flask(__name__)
CORS(app)

def load_image(image):
    image = Image.open(io.BytesIO(image))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def transform_data(rgb_image):
    data_transform = transforms.Compose(
        [
            transforms.Resize((224)),
            transforms.ToTensor()
        ]
    )
    tensor = data_transform(rgb_image)
    result = torch.unsqueeze(tensor, 0)
    return result


def load_model(number_of_classes):
    global model

    model = ResNet_model(number_of_classes)
    model_path = os.getenv("MODEL_PATH")
    model.load_state_dict(torch.load(model_path))
    model.network.fc = nn.Sequential()
    model.eval()
    torch.cuda.set_device(gpu_ids)
    cudnn.benchmark = True
    if torch.cuda.is_available():
        model.cuda()


def extrac_freature(image):
    global model
    image_trans = transform_data(image)
    output = model(image_trans)
    return output


