import requests
from PIL import Image
from io import BytesIO
import tempfile
import torch
from pathlib import Path


def get_img(url):
    response = requests.get(url)
    response.raise_for_status()

    img = Image.open(BytesIO(response.content)).convert("RGB")

    temp_file = tempfile.NamedTemporaryFile(suffix=".jpeg", delete=False)
    img.save(temp_file.name, format="JPEG")

    print(f"Image saved to temporary file: {temp_file.name}")
    return temp_file.name


def save_tensor(img_id, tensor):
    path = Path(f"data/tensors/{img_id}")
    torch.save(tensor.cpu(), f"{path}.pt")


def load_tensor(filename):
    tensor = torch.load(f"data/tensors/{filename}")
    return tensor
