import os
from PIL import Image


def to_png(folder):
    for name in os.listdir(path='./' + folder):
        old_path = folder + '/' + name
        img = Image.open(old_path)
        part, ext = os.path.splitext(old_path)
        new_path = part + '.png'
        img.save(new_path)
        os.remove(old_path)
