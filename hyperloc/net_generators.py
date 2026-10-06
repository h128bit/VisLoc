import timm 
from torch import nn


def generate_resnet18(number: int):
    def get_model():
        model = timm.create_model("resnet18", pretrained=False)
        model.fc = nn.Linear(in_features=512, out_features=2)
        return model

    def generate():
        return [get_model() for _ in range(number)]

    return generate

    
