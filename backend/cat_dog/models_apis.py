import torch
import torch.nn as nn
from PIL import Image
import requests
from io import BytesIO
import torchvision.transforms as T
import numpy as np

############################## PYTORCH MODEL ########################################
# Craete a neural network from pytorch
# https://www.kaggle.com/code/reukki/pytorch-cnn-tutorial-with-cats-and-dogs


class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fc1 = nn.Linear(3*3*64, 10)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(10, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


model = Cnn()
model_loaded = False

# Image transformer!!
image_size = 224
img_transformer = T.Compose([
    # T.ToPILImage(),
    T.Resize((image_size, image_size)),
    T.Grayscale(),
    T.ToTensor()
])

# Encoder decoder


class LabelTransformer():
    labels_map = {}
    labels_id_map = {}

    def __init__(self, labels):
        # Create a labelMap
        labels_set = set(labels)

        for id, val in enumerate(labels_set):
            self.labels_map[val] = id
            self.labels_id_map[id] = val

    def encoder(self, label):
        return self.labels_map[label]

    def decoder(self, label_encoded):
        return self.labels_id_map[label_encoded]


classes = ["cat", "dog"]
label_transformer = LabelTransformer(classes)
############################## PYTORCH MODEL ########################################


def load_model():
    model.eval()
    model.load_state_dict(torch.load(
        "/Users/austonpramodh/Desktop/MyProjects/ml-cat-dog/backend/cat_dog/model-state.pth", map_location=torch.device('cpu')))


def infer_image(img_url):
    if model_loaded != True:
        load_model()

    # Get the image
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content))
    # Transform the image
    transformed_img = img_transformer(img)
    # convert to the required data structure
    transformed_img_tensor_unsqueezed = transformed_img.unsqueeze(0)
    # Infer the image type
    model_response = np.argmax(
        model(transformed_img_tensor_unsqueezed).detach().numpy())
    # Decode the label
    print(label_transformer.decoder(model_response))

    return label_transformer.decoder(model_response)
