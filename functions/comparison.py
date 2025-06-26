import torch.nn.functional as F
import os
from PIL import Image
import torch
from torchvision import models
from torchvision.transforms import transforms
import gdown


def compare(test_tensor, control_tensor):
    """
    input: two tensors
    func:
        - determine similarity
    return : Boolean Value of similarity
    """
    cos_sim = F.cosine_similarity(test_tensor.unsqueeze(0), control_tensor.unsqueeze(0))
    if 0.75 <= cos_sim:
        return True
    return False


def infer(path):
    """
    input: blob storage url
    func:
        - download raw img from url.
        - preprocess raw img to tensor
        - feature extraction using pretrained VGG
    return : tensor of extracted features
    """
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    vgg = models.vgg16(weights=None)
    url = "https://drive.google.com/uc?id=1s0L_4Cl5rEv6wu4pPfS2kygGp7g7S4ky"
    output = "vgg.pth"
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    vgg.load_state_dict(torch.load(output, map_location="cpu", weights_only=False))
    model = vgg.features.eval()

    img = Image.open(path).convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0)  # add batch dim for PyTorch input

    with torch.no_grad():
        feature_map = model(input_tensor)  # (1, 512, 7, 7)
        vector = torch.flatten(feature_map, start_dim=1)  # (1, 25088)

    return vector.squeeze(0)
