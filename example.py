# micro_service.py
import io
import os
import time
import numpy as np
import flask
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from PIL import Image
from flask import Flask, jsonify
from flask_cors import CORS
from torch.autograd import Variable
from torchvision import transforms
from dotenv import load_dotenv

from model import ft_net

env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv(dotenv_path=env_path)


# flask app
app = Flask(__name__)
CORS(app)

# config parameters
gpu_ids = int(os.getenv("GPU_ID"))
nclasses = int(os.getenv("CLASSES"))  # Number of classes in model

# model
model = None


# Load model
def load_network():
    global model

    # load model structure
    model = ft_net(nclasses)
    # add weights into model
    model_path = os.getenv("MODEL_PATH")
    model.load_state_dict(torch.load(model_path))
    # Remove the final fc layer and classifier layer
    model.classifier.classifier = nn.Sequential()
    # Change to test mode
    model = model.eval()
    # set gpu ids
    torch.cuda.set_device(gpu_ids)
    cudnn.benchmark = True
    use_gpu = torch.cuda.is_available()
    # check condition
    if use_gpu:
        model = model.cuda()


# flip image horizontal
def fliplr(img):
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()
    img_flip = img.index_select(3, inv_idx)
    return img_flip


# load image
def load_image(image):
    image = Image.open(io.BytesIO(image))
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


# extract feature
def extract_feature(img):
    global model

    features = torch.FloatTensor()
    ff = torch.FloatTensor(1, 512).zero_().cuda()
    for i in range(2):
        if i == 1:
            img = fliplr(img)
        input_img = Variable(img.cuda())
        outputs = model(input_img)
        ff += outputs
    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
    ff = ff.div(fnorm.expand_as(ff))
    features = torch.cat((features, ff.data.cpu()), 0)
    return features


def extract_feature_batch(img):
    global model

    features = torch.FloatTensor()
    n, c, h, w = img.size()
    ff = torch.FloatTensor(n, 512).zero_().cuda()
    for i in range(2):
        if i == 1:
            img = fliplr(img)
        input_img = Variable(img.cuda())
        outputs = model(input_img)
        ff += outputs
    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
    ff = ff.div(fnorm.expand_as(ff))
    features = torch.cat((features, ff.data.cpu()), 0)
    return features


def transform_data(rgb_image):
    """Transform PIL image to Pytorch Tensor

    Args:
        rgb_image (PIL.image): PIL image type
        torch_tensor:
    """
    data_transform = transforms.Compose(
        [
            transforms.Resize((256, 128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    tensor = data_transform(rgb_image)
    return torch.unsqueeze(tensor, 0)


def warmup():
    """ Warm up process
    - Create a dummy data
    """
    global model

    # create dummy data
    img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    rgb_image = Image.fromarray(img_array)

    # data transformation
    img = transform_data(rgb_image)

    features = torch.FloatTensor()
    ff = torch.FloatTensor(1, 512).zero_().cuda()
    for i in range(2):
        if i == 1:
            img = fliplr(img)
        input_img = Variable(img.cuda())
        outputs = model(input_img)
        ff += outputs
    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
    ff = ff.div(fnorm.expand_as(ff))
    features = torch.cat((features, ff.data.cpu()), 0)

    print("[INFO] Finished warming up")


@app.route("/predict/person_embedding", methods=["POST"])
def reid():
    global model

    # measure performance - starttime
    _start_time = time.time()

    # parse the request
    rgb_image = load_image(flask.request.files["image"].read())

    # rgb_image: PIL Image
    data = transform_data(rgb_image)

    # Extract feature
    with torch.no_grad():
        feature = extract_feature(data)

    print(f"[INFO] Processing time: {time.time() - _start_time}")

    response_data = []

    response_data.append({"embedding": feature.tolist()[0], "height": rgb_image.size[1], "width": rgb_image.size[0]})

    data = {"success": True, "data": response_data}

    return jsonify(data)


@app.route("/predict/person_embedding_batch", methods=["POST"])
def reid_batch():
    global model

    _start_time = time.time()

    # parse the request
    files = flask.request.files.getlist("images")

    tensors = []
    heights = []
    widths = []
    for file in files:
        rgb_image = load_image(file.read())
        heights.append(rgb_image.size[1])
        widths.append(rgb_image.size[0])

        tensors.append(transform_data(rgb_image))

    batch_tensors = torch.cat(tensors, dim=0)

    # Extract feature
    with torch.no_grad():
        feature = extract_feature_batch(batch_tensors)

    print(f"[INFO] Processing time: {time.time() - _start_time}")

    response_data = []

    for vector, height, width in zip(feature.tolist(), heights, widths):
        response_data.append({"embedding": vector, "height": height, "width": width})

    # output data
    data = {"success": True, "data": response_data}
    return jsonify(data)


@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    return "Hello world", 200


if __name__ == "__main__":
    load_network()
    warmup()
    app.run(host=os.getenv("MODEL_HOST"), port=int(os.getenv("MODEL_PORT")), threaded=False, debug=False)
