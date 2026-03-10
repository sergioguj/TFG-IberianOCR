from pathlib import Path
import shutil
import re
import json
import random
import os
import matplotlib
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageFilter, ImageChops, ImageDraw, ImageFont
SHOW_PLOTS = os.environ.get("IBERIAN_SHOW", "0") == "1"

matplotlib.use("TkAgg" if SHOW_PLOTS else "Agg")  


# Rutas base
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "runs" / "dataset_actual"
TRAIN_OUT_DIR = BASE_DIR / "runs" / "exp_actual"          # salida fija de entrenamiento
PREVIEW_DIR   = DATASET_DIR / "preview_images"            # export de imágenes (train/test)

# Parametros por defecto
IMG_SIZE = 28
SAMPLES_PER_CLASS = 1200
TEST_SPLIT = 0.2
ROT_DEG = 12
SHIFT_PX = 3
SCALE_JITTER = 0.10
APPLY_BLUR_P = 0.3
THRESH = 210
USE_FONT_SAMPLES = False
FONT_FRACTION = 0.15

# Clases y extensiones
CLASS_NAMES = [f"simbolo_{i}" for i in range(1, 96)]


CLASS_TO_CHAR_MAP = {
    "simbolo_1": "a",   
    "simbolo_2": "b",
    "simbolo_3": "c",
    "simbolo_4": "e",
    "simbolo_5": "d",
    "simbolo_6": "f",
    "simbolo_7": "g",
    "simbolo_8": "h",
    "simbolo_9": "i",
    "simbolo_10": "j",
    "simbolo_11": "k",
    "simbolo_12": "l",
    "simbolo_13": "m",
    "simbolo_14": "n",
    "simbolo_15": "o",
    "simbolo_16": "p",
    "simbolo_17": "q",
    "simbolo_18": "r",
    "simbolo_19": "s",
    "simbolo_20": "t",
    "simbolo_21": "u",
    "simbolo_22": "v",
    "simbolo_23": "w",
    "simbolo_24": "x",
    "simbolo_25": "y",
    "simbolo_26": "z",
    "simbolo_27": "A",
    "simbolo_28": "B",
    "simbolo_29": "C",
    "simbolo_30": "D",
    "simbolo_31": "E",
    "simbolo_32": "F",
    "simbolo_33": "G",
    "simbolo_34": "H",
    "simbolo_35": "I",
    "simbolo_36": "J",
    "simbolo_37": "K",
    "simbolo_38": "L",
    "simbolo_39": "M",
    "simbolo_40": "N",
    "simbolo_41": "O",
    "simbolo_42": "P",
    "simbolo_43": "Q",
    "simbolo_44": "R",
    "simbolo_45": "S",
    "simbolo_46": "T",
    "simbolo_47": "U",
    "simbolo_48": "V",
    "simbolo_49": "W",
    "simbolo_50": "X",
    "simbolo_51": "Y",
    "simbolo_52": "Z",
    "simbolo_53": "0",
    "simbolo_54": "1",
    "simbolo_55": "2",
    "simbolo_56": "3",
    "simbolo_57": "4",
    "simbolo_58": "5",
    "simbolo_59": "6",
    "simbolo_60": "7",
    "simbolo_61": "8",
    "simbolo_62": "9",
    "simbolo_63": "!",
    "simbolo_64": "\"",
    "simbolo_65": "#",
    "simbolo_66": "$",
    "simbolo_67": "%",
    "simbolo_68": "&",
    "simbolo_69": "'",
    "simbolo_70": "(",
    "simbolo_71": ")",
    "simbolo_72": "*",
    "simbolo_73": "+",
    "simbolo_74": ",",
    "simbolo_75": "-",
    "simbolo_76": ".",
    "simbolo_77": "/",
    "simbolo_78": ":",
    "simbolo_79": ";",
    "simbolo_80": "<",
    "simbolo_81": "=",
    "simbolo_82": ">",
    "simbolo_83": "?",
    "simbolo_84": "@",
    "simbolo_85": "[",
    "simbolo_86": "\\", 
    "simbolo_87": "]",
    "simbolo_88": "^",
    "simbolo_89": "_",
    "simbolo_90": "",
    "simbolo_91": "",
    "simbolo_92": "",
    "simbolo_93": "",
    "simbolo_94": "",
    "simbolo_95": "",
}
EXTS = {".png", ".jpg", ".jpeg"}

# Matplotlib
mpl.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
})

# Utilidades de FS
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def clear_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

def save_json(d: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

def savefig(fig, out_dir: Path, name: str, show: bool | None = None):
    if show is None:
        from common import SHOW_PLOTS 
        show = SHOW_PLOTS
    ensure_dir(out_dir)
    path = out_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return str(path)

# Escaneo de seedds
def natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]

def scan_seed_images(seeds_dir: Path):
    seeds_dir = Path(seeds_dir)
    class_images = {
        cls: sorted(
            [p for p in (seeds_dir / cls).glob("*") if p.is_file() and p.suffix.lower() in EXTS],
            key=natural_key
        )
        for cls in CLASS_NAMES
    }
    missing = [cls for cls, files in class_images.items() if not files]
    return class_images, missing

# Imagen y augment
def _to_square(img: Image.Image, size=IMG_SIZE, inner=22):
    img = ImageOps.contain(img, (inner, inner), Image.BICUBIC)
    canvas = Image.new("L", (size, size), 255)
    x = (size - img.width) // 2
    y = (size - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas

def load_seed_gray(path: str, thresh=THRESH, size=IMG_SIZE):
    img = Image.open(path).convert("L")
    arr = np.array(img)
    if arr.mean() < 128:
        img = ImageOps.invert(img)
    arr = np.where(np.array(img) < thresh, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    bbox = ImageOps.invert(img).getbbox() or (0, 0, *img.size)
    img = img.crop(bbox)
    return _to_square(img, size=size)

def augment_28(img: Image.Image, rot=ROT_DEG, shift=SHIFT_PX, jitter=SCALE_JITTER, blur_p=APPLY_BLUR_P):
    scale = 1.0 + random.uniform(-jitter, jitter)
    new_w = max(6, int(img.width * scale))
    new_h = max(6, int(img.height * scale))
    aug = img.resize((new_w, new_h), Image.BICUBIC)

    angle = random.uniform(-rot, rot)
    aug = aug.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=255)

    dx = random.randint(-shift, shift)
    dy = random.randint(-shift, shift)
    aug = ImageChops.offset(aug, dx, dy)
    d = ImageDraw.Draw(aug)
    if dx > 0: d.rectangle((0, 0, dx, aug.height), fill=255)
    if dx < 0: d.rectangle((aug.width + dx, 0, aug.width, aug.height), fill=255)
    if dy > 0: d.rectangle((0, 0, aug.width, dy), fill=255)
    if dy < 0: d.rectangle((0, aug.height + dy, aug.width, aug.height), fill=255)

    if random.random() < blur_p:
        aug = aug.filter(ImageFilter.GaussianBlur(radius=0.6))

    bbox = ImageOps.invert(aug).getbbox()
    if bbox:
        aug = aug.crop(bbox)
    return _to_square(aug, size=IMG_SIZE)

def render_from_font_28(char: str, font_path: str):
    font = ImageFont.truetype(font_path, 22)
    tmp = Image.new("L", (60, 60), 255)
    d = ImageDraw.Draw(tmp)
    bbox = font.getbbox(char)
    x = (tmp.width - (bbox[2] - bbox[0])) // 2 - bbox[0]
    y = (tmp.height - (bbox[3] - bbox[1])) // 2 - bbox[1]
    d.text((x, y), char, font=font, fill=0)
    bbox = ImageOps.invert(tmp).getbbox()
    if bbox:
        tmp = tmp.crop(bbox)
    return _to_square(tmp, size=IMG_SIZE)

# Visualizaciones
def ver_ejemplos(X, y, class_names, out_dir: Path, n=40, cols=10, cell=1.8, title="Muestras"):
    n = min(n, len(X))
    idxs = np.random.choice(len(X), size=n, replace=False)
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(cols*cell, rows*cell))
    for i, idx in enumerate(idxs):
        ax = fig.add_subplot(rows, cols, i+1)
        ax.imshow(X[idx], cmap="gray")
        ax.set_title(class_names[int(y[idx])], fontsize=8)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return savefig(fig, out_dir, "ejemplos")
