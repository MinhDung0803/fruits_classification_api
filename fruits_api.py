import torch
import os
from dotenv import load_dotenv
from models import ResNet_model
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv(dotenv_path=env_path)

# config parameters
gpu_ids = int(os.getenv("GPU_ID"))
nclasses = int(os.getenv("CLASSES"))  # Number of classes in model

# model
model = None

def transform_data(rgb_image):
    data_transform = transforms.Compose(
        [
            transforms.Resize((224)),
            transforms.ToTensor()
        ]
    )
    tensor = data_transform(rgb_image)
    print(tensor.shape)
    result = torch.unsqueeze(tensor, 0)
    print(result.shape)
    return result


def load_model(num_of_classes):
    global model

    # get model architecture
    model = ResNet_model(num_of_classes)
    model_path = os.getenv("MODEL_PATH")
    model = torch.load(model_path)
    # Remove the final fc layer and classifier layer
    model.network.fc = nn.Sequential()
    # Change to test mode
    model = model.eval()
    # set gpu ids
    torch.cuda.set_device(gpu_ids)
    # inbuilt cudnn auto-tuner to find the best algorithm with fixed input size
    cudnn.benchmark = True
    use_gpu = torch.cuda.is_available()
    # check condition
    if use_gpu:
        model = model.cuda()


def warmup():
    """ Warm up process
    - Create a dummy data
    """
    global model

    # create dummy data
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    rgb_image = Image.fromarray(img_array)

    # data transformation
    img = transform_data(rgb_image)

    # features = torch.FloatTensor()
    # ff = torch.FloatTensor(1, 512).zero_().cuda()
    input_img = Variable(img.cuda())
    # print(input_img.shape)
    outputs = model(input_img)
    print(outputs.shape)
    print(outputs)


    print("[INFO] Finished warming up")

# load_model(nclasses)
# warmup()