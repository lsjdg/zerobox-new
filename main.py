from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from functions.storage import *
from functions.comparison import *
from functions.blur import *


app = FastAPI()


class Image(BaseModel):
    id: int
    url: str


@app.post("/inference")
async def infer_image(image: Image):
    temp_path = get_img(image.url)
    vector = infer(temp_path)
    save_tensor(image.id, vector)
    os.remove(temp_path)


@app.post("/inspect")
def inspect_image(image: Image):
    temp_path = get_img(image.url)

    is_similar = 0
    dir_path = "data/tensors"
    file_names = [
        f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))
    ]
    vector = infer(temp_path)
    for filename in file_names:
        vector_compare = load_tensor(filename)
        if compare(vector, vector_compare):
            is_similar = 1
            break

    is_blurry = inspect_quality(temp_path)
    result = is_similar * 2 + is_blurry

    os.remove(temp_path)

    return result


if __name__ == "__main__":
    uvicorn.run("API.inspection_app:app", host="0.0.0.0", port=8000, reload=True)
