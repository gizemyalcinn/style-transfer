import torch
from torchvision.models import vgg19, VGG19_Weights

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

LAYERS = {1: "relu1_1", 6: "relu2_1", 11: "relu3_1",
          20: "relu4_1", 22: "relu4_2", 29: "relu5_1"}
STYLE_LAYERS = ["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"]
CONTENT_LAYER = "relu4_2"

def load_vgg(device):
    vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:30].to(device).eval()
    for p in vgg.parameters():
        p.requires_grad_(False)
    return vgg

def get_features(image, model):
    x = (image - MEAN.to(image.device)) / STD.to(image.device)
    features = {}
    for i, layer in enumerate(model):
        x = layer(x)
        if i in LAYERS:
            features[LAYERS[i]] = x
    return features

def gram_matrix(x):
    B, C, H, W = x.shape
    F = x.view(B, C, H * W)
    return (F @ F.transpose(1, 2)) / (C * H * W)