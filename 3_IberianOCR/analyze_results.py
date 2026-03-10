import os
os.environ["IBERIAN_SHOW"] = "1"  

import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from common import CLASS_NAMES, ensure_dir, savefig, TRAIN_OUT_DIR


SKLEARN_OK = True
try:
    from sklearn.metrics import confusion_matrix, classification_report
except Exception:
    SKLEARN_OK = False

def plot_curvas(history, out_dir: Path):
    fig1 = plt.figure(figsize=(6,4))
    plt.plot(history.get("loss", []), label="train_loss")
    if "val_loss" in history: plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Pérdida"); plt.legend(); plt.title("Evolución de la pérdida"); plt.grid(True, alpha=0.3)
    fig1.tight_layout()
    savefig(fig1, out_dir, "curva_perdida")

    fig2 = plt.figure(figsize=(6,4))
    plt.plot(history.get("accuracy", []), label="train_acc")
    if "val_accuracy" in history: plt.plot(history["val_accuracy"], label="val_acc")
    plt.xlabel("Epoch"); plt.ylabel("Precisión"); plt.legend(); plt.title("Evolución de la precisión"); plt.grid(True, alpha=0.3)
    fig2.tight_layout()
    savefig(fig2, out_dir, "curva_precision")

def plot_confusion_matrix_large(cm, class_names, out_dir: Path, normalize=True, name="matriz_confusion_normalizada"):
    cm_plot = cm.astype(np.float32)
    if normalize:
        row_sums = cm_plot.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_plot = cm_plot / row_sums
    n = len(class_names)
    fig = plt.figure(figsize=(min(28, max(12, n*0.22)), min(28, max(12, n*0.22))))
    plt.imshow(cm_plot, interpolation="nearest")
    plt.title("Matriz de confusión" + (" (normalizada)" if normalize else ""))
    plt.xlabel("Predicción"); plt.ylabel("Etiqueta real")
    cb = plt.colorbar(); cb.ax.tick_params(labelsize=9)
    step = max(1, n // 20)
    ticks = np.arange(0, n, step)
    plt.xticks(ticks, ticks, rotation=90)
    plt.yticks(ticks, ticks)
    plt.tight_layout()
    savefig(fig, out_dir, name)

def top_confusions(cm, class_names, out_dir: Path, topn=30, min_count=5, name="top_confusiones"):
    cm2 = cm.copy()
    np.fill_diagonal(cm2, 0)
    pairs = []
    for i in range(cm2.shape[0]):
        for j in range(cm2.shape[1]):
            c = int(cm2[i, j])
            if c >= min_count:
                pairs.append((c, i, j))
    pairs.sort(reverse=True, key=lambda x: x[0])
    pairs = pairs[:topn]
    if not pairs:
        print("No hay confusiones significativas según el umbral.")
        return
    fig = plt.figure(figsize=(10, max(5, len(pairs)*0.4)))
    labels = [f"{class_names[i]} → {class_names[j]}" for _, i, j in pairs]
    values = [c for c, _, _ in pairs]
    y = np.arange(len(labels))
    plt.barh(y, values)
    plt.yticks(y, labels)
    plt.xlabel("Recuentos de confusión"); plt.title("Top confusiones (off-diagonal)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    savefig(fig, out_dir, name)

def per_class_accuracy(y_true, y_pred, class_names, out_dir: Path, per_page=32, label_mode="idx", name="accuracy_por_clase"):
    n = len(class_names)
    accs = np.zeros(n, dtype=np.float32)
    for c in range(n):
        m = (y_true == c)
        total = m.sum()
        accs[c] = (y_pred[m] == y_true[m]).mean() if total > 0 else np.nan

    order = np.argsort(accs)
    pages = int(np.ceil(n / per_page))
    for p in range(pages):
        sel = order[p*per_page:(p+1)*per_page]
        vals = np.nan_to_num(accs[sel])
        labels = [str(i) for i in sel] if label_mode=="idx" else [f"{class_names[i]} ({i})" for i in sel]
        h = max(5, 0.36 * len(sel))
        fig = plt.figure(figsize=(11, h))
        y = np.arange(len(sel))
        plt.barh(y, vals)
        plt.yticks(y, labels, fontsize=8 if label_mode=="full" else 9)
        plt.xlabel("Accuracy por clase")
        plt.title(f"Accuracy por clase (ordenado) — página {p+1}/{pages}")
        plt.gca().invert_yaxis()
        left_margin = 0.30 if label_mode == "full" else 0.12
        plt.tight_layout(); plt.subplots_adjust(left=left_margin, right=0.98, top=0.92)
        for yi, v in enumerate(vals):
            plt.text(v + 0.005, yi, f"{v:.2f}", va="center", fontsize=7)
        savefig(fig, out_dir, f"{name}_p{p+1}")

def main():
    ap = argparse.ArgumentParser(description="Generar gráficas e informes desde artefactos guardados")
    ap.add_argument("--run-dir", type=Path, default=TRAIN_OUT_DIR, help="Carpeta con history.json, y_pred_test.npy, etc. (por defecto runs/exp_actual/)")
    ap.add_argument("--also-samples", dest="also_samples", action="store_true", help="Dibujar muestras por clase con el modelo")
    args = ap.parse_args()

    run = Path(args.run_dir)
    plots_dir = run / "plots"
    ensure_dir(plots_dir)

    hist_path = run / "history.json"
    if hist_path.exists():
        history = json.loads(hist_path.read_text(encoding="utf-8"))
        plot_curvas(history, plots_dir)

    y_pred_path = run / "y_pred_test.npy"
    test_y_path = run / "test_y.npy"

    if y_pred_path.exists() and test_y_path.exists() and SKLEARN_OK:
        y_pred = np.load(y_pred_path)
        test_y = np.load(test_y_path)
        cm = confusion_matrix(test_y, y_pred)
        print("\nReporte de clasificación:\n", classification_report(test_y, y_pred, digits=4))
        plot_confusion_matrix_large(cm, CLASS_NAMES, plots_dir, normalize=True, name="matriz_confusion_normalizada")
        top_confusions(cm, CLASS_NAMES, plots_dir, topn=30, min_count=5, name="top_confusiones")
        per_class_accuracy(test_y, y_pred, CLASS_NAMES, plots_dir, per_page=32, label_mode="idx")

    if args.also_samples:
        import tensorflow as tf
        test_X_path = run / "test_X.npy"
        model_path  = run / "iberian_mlp_demo.keras"
        if test_X_path.exists() and test_y_path.exists() and model_path.exists():
            test_X = np.load(test_X_path).astype("float32") / 255.0
            test_y = np.load(test_y_path)
            model = tf.keras.models.load_model(model_path)

            indices_por_clase = []
            for c in range(len(CLASS_NAMES)):
                where = np.where(test_y == c)[0]
                if len(where) > 0:
                    indices_por_clase.append(where[0])

            if indices_por_clase:
                imgs = test_X[indices_por_clase]
                probs = model.predict(imgs, verbose=0)
                y_pred_small = probs.argmax(axis=1)
                confs = probs.max(axis=1)

                n = len(indices_por_clase); cols=10; cell=1.9
                per_page = cols*7
                pages = int(np.ceil(n / per_page))
                for p in range(pages):
                    chunk = slice(p*per_page, (p+1)*per_page)
                    ids = indices_por_clase[chunk]
                    rows = int(np.ceil(len(ids) / cols))
                    fig = plt.figure(figsize=(cols*cell, rows*(cell+0.3)))
                    for i, idx in enumerate(ids):
                        ax = fig.add_subplot(rows, cols, i+1)
                        ax.imshow(test_X[idx], cmap="gray")
                        pr = int(y_pred_small[p*per_page+i]); gt = int(test_y[idx]); conf=float(confs[p*per_page+i])
                        ax.set_title(f"{gt}→{pr} | {conf:.2f}", fontsize=7, pad=2)
                        ax.axis("off")
                    fig.suptitle(f"Muestra por clase (página {p+1}/{pages})", fontsize=11)
                    fig.tight_layout(); fig.subplots_adjust(hspace=0.6, top=0.92)
                    savefig(fig, plots_dir, f"muestra_por_clase_p{p+1}")

if __name__ == "__main__":
    main()
