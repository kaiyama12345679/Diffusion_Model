import os
import math
import random
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
import os
import json
from dataclasses import dataclass, field

# 漢字の画像を生成するため
from fontTools import ttLib
from PIL import Image, ImageFont, ImageDraw

def create_kanji_images(pix: int, font_file: str):

    image_size = (pix, pix)
    fontsize = pix - 2 * 2
    margin = (2, 2)
    image_font = ImageFont.truetype(font=font_file, size=fontsize, index=0)
    with ttLib.TTFont(font_file, fontNumber=0) as font:
        cmap = font.getBestCmap()

    arr_list = []
    for cid in range(0x4E00, 0x9FFF):
        if cid not in cmap.keys():
            continue

        letter = chr(cid)
        img = Image.new("L", image_size, color=0)
        draw = ImageDraw.Draw(img)
        draw.text(margin, letter, font=image_font, fill=1)
        arr_list.append(np.array(img)[np.newaxis, :, :])
    return np.concatenate(arr_list, axis=0)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, pix, font_file):
        img_np = create_kanji_images(pix, font_file)
        self.img = torch.from_numpy(img_np).float()
        self.img = self.img.unsqueeze(1)

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        x = self.img[idx, ...]
        return x

if __name__ == "__main__":
    font_file = ".src//gomarice_mukasi_mukasi.ttf"
    pix = 128
    dataset = Dataset(pix, font_file)
    img = dataset[10]
    # save image
    img = img.squeeze().numpy()
    print(np.max(img), np.min(img))