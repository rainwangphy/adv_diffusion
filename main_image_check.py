from pathlib import Path
import os

data_dir = "./images"
exts = ["jpg", "jpeg", "png", "tiff"]
paths = [p for ext in exts for p in Path(f"{data_dir}").glob(f"**/*.{ext}")]

print(paths)
from PIL import Image

for index in range(len(paths)):
    path = paths[index]
    img = Image.open(path)
    if len(img.split()) != 3:
        print("caught")
        os.remove(path)
