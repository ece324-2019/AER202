import os
import torch
import torch.nn as nn
import torchvision
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image

MODEL_DIR = '../models/'
MANUAL_CHECK_DIR = '../datasets/manual_check/'
RESIZED_IMAGE_DIM = 224
SEED = 1
NUM_CLASSES = 12


def build_classifier(num_in_features, hidden_layers, num_out_features):
    classifier = nn.Sequential()

    # NO HIDDEN LAYERS
    if hidden_layers == None:
        classifier.add_module('fc0', nn.Linear(num_in_features, num_out_features))

    # HAVE HIDDEN LAYERS
    else:
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        # Fully connected layers to be optimized
        classifier.add_module('fc0', nn.Linear(num_in_features, hidden_layers[0]))
        classifier.add_module('relu0', nn.ReLU())
        # Dropout to reduce overfitting
        classifier.add_module('drop0', nn.Dropout(.6))

        for i, (h1, h2) in enumerate(layer_sizes):
            classifier.add_module('fc' + str(i + 1), nn.Linear(h1, h2))
            classifier.add_module('relu' + str(i + 1), nn.ReLU())
            classifier.add_module('drop' + str(i + 1), nn.Dropout(.5))
        classifier.add_module('output', nn.Linear(hidden_layers[-1], num_out_features))

    return classifier


def load_saved_model(saved_model_name: str):
    """Load a saved neural network"""

    saved_model = models.resnet152(pretrained=True)
    saved_model.state_dict(torch.load(MODEL_DIR + saved_model_name))
    num_in_features = 2048
    hidden_layers = None
    classifier = build_classifier(num_in_features, hidden_layers, NUM_CLASSES)
    saved_model.fc = classifier

    for parameter in saved_model.parameters():
        parameter.requires_grad = False

    saved_model.eval()

    return saved_model


def load_image(image_path: str):
    image = Image.open(image_path)
    processed_image = image.resize((RESIZED_IMAGE_DIM, RESIZED_IMAGE_DIM))

    # Convert image to numpy array
    np_arr = np.array(processed_image).astype('float32')
    # Normalize image
    for i in range(3):
        minval = np_arr[..., i].min()
        maxval = np_arr[..., i].max()
        if minval != maxval:
            np_arr[..., i] -= minval
            # np_arr[..., i] *= (255.0 / (maxval - minval))
            np_arr[..., i] /= 255.0
    # normalized_image = Image.fromarray(np_arr.astype('uint8'), 'RGB')
    # normalized_image.save(MANUAL_CHECK_DIR + 'test.jpg')

    # return torch.from_numpy(np.array(processed_image).astype('float32'))
    return torch.from_numpy(np_arr)


def manual_check(saved_model_name: str, image_path: str):
    model = load_saved_model(saved_model_name)
    # model = model.to('cpu')
    image = load_image(image_path)
    # print('image:', image.type())
    classes = (
        'Air Canada Airbus A319',
        'Air Canada Airbus A320',
        'Air Canada Airbus A321',
        'Air Canada Airbus A330-300',
        'Air Canada Boeing 767-300',
        'Air Canada Rouge Boeing 767-300',
        'Air Transat Airbus A330-300',
        'Airbus Beluga',
        'American Boeing 777-300ER',
        'Lufthansa Boeing 747-400',
        'Porter Bombardier Q400',
        'Westjet Boeing 737-800'
    )
    image = image.unsqueeze(0)
    image = image.permute(0, 3, 1, 2)
    print('image.mean:', image.mean())
    print('image.std:', image.std())
    plt.imshow(image.squeeze().permute(1, 2, 0))
    plt.show()
    prediction = model(image)
    _, preds = torch.max(prediction, 1)
    print('prediction:', classes[preds.item()])


manual_check('aer202_resnet_model_2.pt', MANUAL_CHECK_DIR + '132.jpg')
