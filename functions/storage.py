import requests
from PIL import Image
from io import BytesIO
import tempfile
import torch
from pathlib import Path

PERSIST_ROOT = Path("/home/data")
TENSOR_DIR = PERSIST_ROOT / "tensors"
TENSOR_DIR.mkdir(parents=True, exist_ok=True)


def get_img(url):
    response = requests.get(url)
    response.raise_for_status()

    img = Image.open(BytesIO(response.content)).convert("RGB")

    temp_file = tempfile.NamedTemporaryFile(suffix=".jpeg", delete=False)
    img.save(temp_file.name, format="JPEG")

    print(f"Image saved to temporary file: {temp_file.name}")
    return temp_file.name


def save_tensor(img_id, tensor):
    torch.save(tensor.cpu(), TENSOR_DIR / f"{img_id}.pt")
    print(f"tensor saved for {img_id}")


def load_tensor(filename):
    return torch.load(TENSOR_DIR / filename, map_location="cpu")
