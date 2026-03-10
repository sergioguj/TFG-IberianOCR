import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense
from common import CLASS_NAMES, ensure_dir, save_json, DATASET_DIR, TRAIN_OUT_DIR

# En train_model.py

from tensorflow.keras.layers import Input, Flatten, Dense, Dropout

def build_mlp(num_classes):
    model = Sequential([
        Input(shape=(28, 28)),
        Flatten(),

        Dense(128, activation="relu", name="layer1"),    
        Dropout(0.3), 
        Dense(64, activation="relu", name="layer2"),
        
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def main():
    ap = argparse.ArgumentParser(description="Entrenar MLP con dataset guardado (salida fija en runs/exp_actual/)")
    # Dataset fijo por defecto 
    ap.add_argument("--data-dir", type=Path, default=DATASET_DIR,
                    help="Carpeta del dataset (por defecto runs/dataset_actual)")
    ap.add_argument("--epochs",   type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    # Rutas fijas
    data_dir = Path(args.data_dir)  # normalmente DATASET_DIR
    out_dir  = TRAIN_OUT_DIR        # SIEMPRE la misma carpeta fija
    plots_dir = out_dir / "plots"
    ensure_dir(out_dir); ensure_dir(plots_dir)

    # Carga de datos
    train_X = np.load(data_dir / "train_X.npy")
    train_y = np.load(data_dir / "train_y.npy")
    test_X  = np.load(data_dir / "test_X.npy")
    test_y  = np.load(data_dir / "test_y.npy")

    train_X = train_X.astype("float32") / 255.0
    test_X  = test_X.astype("float32") / 255.0

    # Modelo
    model = build_mlp(num_classes=len(CLASS_NAMES))
    model.summary()

    # Monitoriza 'val_loss', espera 10 epochs (patience) y restaura el mejor modelo
    early_stopper = EarlyStopping(
        monitor='val_loss',
        patience=10, 
        restore_best_weights=True
    )

    # Entrenamiento
    history = model.fit(
        x=train_X, y=train_y,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
        callbacks=[early_stopper]
    )

    # Guardado artefactos
    hist_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
    save_json(hist_dict, out_dir / "history.json")

    test_loss, test_acc = model.evaluate(test_X, test_y, verbose=0)
    metrics = {"test_loss": float(test_loss), "test_accuracy": float(test_acc)}
    save_json(metrics, out_dir / "metrics.json")

    probs = model.predict(test_X, verbose=0)
    y_pred = probs.argmax(axis=1)
    np.save(out_dir / "y_pred_test.npy", y_pred)
    np.save(out_dir / "probs_test.npy",  probs)

    model_path = out_dir / "iberian_mlp_demo.keras"
    model.save(model_path)
    print(f"Modelo guardado en {model_path}")
    print(f"Artefactos guardados en {out_dir}")
    
    from shutil import copy2
    try:
        src_tx = data_dir / "test_X.npy"
        src_ty = data_dir / "test_y.npy"
        if src_tx.exists():
            copy2(src_tx, out_dir / "test_X.npy")
        if src_ty.exists():
            copy2(src_ty, out_dir / "test_y.npy")
        print("Copiados test_X.npy y test_y.npy a", out_dir)
    except Exception as e:
        print("[AVISO] No se pudieron copiar test_X/test_y al run dir:", e)


if __name__ == "__main__":
    main()
