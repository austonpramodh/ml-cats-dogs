import torch.nn as nn
from PIL import Image
import requests
from io import BytesIO
import torchvision.transforms as T
import numpy as np
import os
import onnxruntime as ort
from sklearn.preprocessing import LabelEncoder

class CustomLabelEncoder:
    def __init__(self):
        self.encoder = LabelEncoder()
    
    def fit_transform(self, labels):
        return self.encoder.fit_transform(labels)
    
    def transform(self, labels):
        return self.encoder.transform(labels)
    
    def inverse_transform(self, encoded_labels):
        return self.encoder.inverse_transform(encoded_labels)

model_onnx_ort_session = None

class ModelData(object):
    # Singleton class
    __instance = None
    is_loaded = False

    def __init__(self):
        if ModelData.__instance != None:
            raise Exception("This class is a singleton! use get_instance()")
        else:
            ModelData.__instance = self
        self.ort_session = None
        self.is_loaded = False
    
    @staticmethod
    def get_instance():
        if ModelData.__instance == None:
            ModelData()
        return ModelData.__instance
    
    def set_session(self, ort_session):
        self.ort_session = ort_session
        self.is_loaded = True

    def get_model_onnx_ort_session(self):
        return self.ort_session

# Image transformer!!
image_size = 224
img_transformer = T.Compose([
    # T.ToPILImage(), # Already a PIL image
    T.Resize((image_size, image_size)),
    T.Grayscale(),
    T.ToTensor()
])

# Encoder decoder

classes = ["cat", "dog"]
encoder = CustomLabelEncoder()
encoded_labels = encoder.fit_transform(classes)

############################## PYTORCH MODEL ########################################


def load_model():
    model_data = ModelData.get_instance()
    # Onnx model
    model_file_path = os.path.join(os.path.dirname(__file__), "model.onnx")
    # load the model
    model_onnx_ort_session = ort.InferenceSession(model_file_path, providers=['CPUExecutionProvider'])
    model_data.set_session(model_onnx_ort_session)
    print("Onxx Model loaded!!")
    


def infer_image(img):
    # Check if img is PIL image
    if not isinstance(img, Image.Image):
        raise Exception("Image should be a PIL image!")
    
    model_data = ModelData.get_instance()
    if not model_data.is_loaded:
        load_model()

    # Transform the image
    transformed_img = img_transformer(img)
    # convert to the required data structure
    transformed_img_tensor_unsqueezed = transformed_img.unsqueeze(0)
    # Infer the image type using onnx model
    # onnx_input = onnx_program.adapt_torch_inputs_to_onnx(torch_input)
    model_data = ModelData.get_instance()
    session = model_data.get_model_onnx_ort_session()
    input_name = session.get_inputs()[0].name
    transformed_img_tensor_unsqueezed = transformed_img_tensor_unsqueezed.detach().cpu().numpy() if transformed_img_tensor_unsqueezed.requires_grad else transformed_img_tensor_unsqueezed.cpu().numpy()
    output = session.run(None, {input_name: transformed_img_tensor_unsqueezed})[0]
    print("ONNX Model!", output)
    model_response_onnx = np.argmax(output)
    print("Onnx model response: ", model_response_onnx)
    print("Onnx model response: ", encoder.inverse_transform([model_response_onnx])[0])

    data = {
        "prediction": encoder.inverse_transform([model_response_onnx])[0],
        "probabilities": {encoder.inverse_transform([i])[0]: round(output[0][i].item(), 4) for i in range(len(output[0]))}
    }

    return data


def infer_image_url(img_url):
    # Get the image
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content))
    return infer_image(img)