import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from common import (
    CLASS_NAMES, scan_seed_images, load_seed_gray, augment_28, render_from_font_28,
    IMG_SIZE, SAMPLES_PER_CLASS, TEST_SPLIT, USE_FONT_SAMPLES, FONT_FRACTION,
    ver_ejemplos, ensure_dir, DATASET_DIR, clear_dir, PREVIEW_DIR
)

def build_dataset(seeds_dir: Path, font_path: Path,
                  samples_per_class=SAMPLES_PER_CLASS, test_split=TEST_SPLIT,
                  use_font=USE_FONT_SAMPLES, font_fraction=FONT_FRACTION,
                  export_images=False, export_format="jpg"):
    out_dir = DATASET_DIR
    clear_dir(out_dir)
    plots_dir = out_dir / "plots"
    ensure_dir(plots_dir)

    class_images, missing = scan_seed_images(seeds_dir)
    if missing:
        print("Aviso: clases sin imágenes ->", ", ".join(missing))
    else:
        print("Todas las clases tienen al menos una imagen.")

    X_list, y_list = [], []
    rng = np.random.default_rng(42)
    use_font = use_font and font_path and Path(font_path).is_file()

    for label, cls in enumerate(CLASS_NAMES):
        paths = class_images.get(cls, [])
        if not paths:
            continue
        seeds = [load_seed_gray(str(p)) for p in paths]
        n_total = samples_per_class
        n_font  = int(n_total * font_fraction) if use_font else 0
        n_png   = n_total - n_font

        for _ in range(n_png):
            base = seeds[rng.integers(0, len(seeds))]
            X_list.append(np.array(augment_28(base), dtype=np.uint8))
            y_list.append(label)

        if use_font:
     
            char_for_class = "X"
            for _ in range(n_font):
                img = render_from_font_28(char_for_class, str(font_path))
                X_list.append(np.array(augment_28(img), dtype=np.uint8))
                y_list.append(label)

    X = np.stack(X_list, axis=0) if X_list else np.empty((0, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    y = np.array(y_list, dtype=np.int64)

    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    n_test = int(len(X) * test_split)
    test_X, test_y = X[:n_test], y[:n_test]
    train_X, train_y = X[n_test:], y[n_test:]

    np.save(out_dir / "train_X.npy", train_X)
    np.save(out_dir / "train_y.npy", train_y)
    np.save(out_dir / "test_X.npy",  test_X)
    np.save(out_dir / "test_y.npy",  test_y)
    
    if export_images:
        clear_dir(PREVIEW_DIR)
        train_dir = PREVIEW_DIR / "train"
        test_dir  = PREVIEW_DIR / "test"
        ensure_dir(train_dir); ensure_dir(test_dir)

        _export_split_to_images(train_X, train_y, CLASS_NAMES, train_dir, fmt=export_format)
        _export_split_to_images(test_X,  test_y,  CLASS_NAMES, test_dir,  fmt=export_format)
        print(f"Imágenes exportadas en: {PREVIEW_DIR} (formato: {export_format})")

    print(f"Dataset guardado en: {out_dir}")
    print("Shapes:",
          "train_X", train_X.shape, "train_y", train_y.shape,
          "test_X",  test_X.shape,  "test_y",  test_y.shape)

    ver_ejemplos(train_X, train_y, CLASS_NAMES, plots_dir, n=40, cols=10, cell=1.8, title="Muestras de entrenamiento")

def _export_split_to_images(split_X, split_y, class_names, out_root, fmt="jpg", quality=90):
    """
    Exporta un split (train o test) a carpetas por clase en out_root.
    split_X: np.ndarray [N, H, W] en uint8
    split_y: np.ndarray [N]
    fmt: 'jpg' | 'png' | 'webp' ...
    """
    ensure_dir(out_root)
    # crear subcarpetas por clase
    for cls in class_names:
        ensure_dir(out_root / cls)

    # contador por clase para nombres consecutivos
    counters = {i: 0 for i in range(len(class_names))}
    for i in range(len(split_X)):
        label = int(split_y[i])
        cls_name = class_names[label]
        im = Image.fromarray(split_X[i], mode="L")
        idx = counters[label]
        counters[label] += 1
        fname = f"{cls_name}_{idx:05d}.{fmt.lower()}"
        save_kwargs = {}
        if fmt.lower() in ("jpg", "jpeg", "webp"):
            save_kwargs["quality"] = quality
        im.save(out_root / cls_name / fname, **save_kwargs)

def main():
    ap = argparse.ArgumentParser(description="Preparar dataset desde seeds/ (guarda en runs/dataset_actual/)")
    ap.add_argument("--seeds-dir", type=Path, default=Path("seeds"))
    ap.add_argument("--font-path", type=Path, default=Path("iberian.ttf"))
    ap.add_argument("--samples-per-class", type=int, default=SAMPLES_PER_CLASS)
    ap.add_argument("--test-split", type=float, default=TEST_SPLIT)
    ap.add_argument("--use-font", action="store_true")
    ap.add_argument("--font-fraction", type=float, default=FONT_FRACTION)
    ap.add_argument("--export-images", action="store_true", help="Exporta train/test como imágenes por clase en runs/dataset_actual/preview_images/")
    ap.add_argument("--export-format", type=str, default="jpg", choices=["jpg", "jpeg", "png", "webp"], help="Formato de exportación de imágenes")

    args = ap.parse_args()

    build_dataset(args.seeds_dir, args.font_path,
                  samples_per_class=args.samples_per_class,
                  test_split=args.test_split,
                  use_font=args.use_font,
                  font_fraction=args.font_fraction,
                  export_images=args.export_images,
                  export_format=args.export_format
                )

if __name__ == "__main__":
    main()
