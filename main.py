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

    file_names = [f for f in TENSOR_DIR.iterdir() if f.is_file()]
    vector = infer(temp_path)

    is_similar = 0
    for f in file_names:
        if compare(vector, load_tensor(f.name)):
            is_similar = 1
            break

    is_blurry = inspect_quality(temp_path)
    result = is_similar * 2 + is_blurry

    os.remove(temp_path)
    return result


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
