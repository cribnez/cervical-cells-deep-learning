"""
Two-Phase Training Pipeline for Cervical Cytology Classification (Bethesda System).
"""
import os
import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import recall_score, precision_score, f1_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from .config import (
    CLASES,
    IMG_SIZE,
    BATCH_SIZE,
    EPOCHS_FASE1,
    EPOCHS_FASE2,
    LR_INICIAL,
    LR_FINE,
    UNFREEZE_LAST_N,
    LABEL_SMOOTHING,
    CLASS_WEIGHT,
    RANDOM_STATE,
    DEFAULT_DATA_DIR,
    DEFAULT_RESULTS_DIR,
)
from .dataset import seed_everything, preparar_dataset_splits
from .preprocessing import preprocess_train_morph, preprocess_eval_morph
from .model import crear_modelo_sequential, guardar_modelo_pickle


def parse_args():
    parser = argparse.ArgumentParser(description="Train EfficientNetB0 for Bethesda Cytology Classification")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR),
                        help="Path to dataset containing class subfolders (HSIL, LSIL, NILM)")
    parser.add_argument("--results_dir", type=str, default=str(DEFAULT_RESULTS_DIR),
                        help="Root directory for saving experiments and model outputs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs_fase1", type=int, default=EPOCHS_FASE1)
    parser.add_argument("--epochs_fase2", type=int, default=EPOCHS_FASE2)
    parser.add_argument("--lr_inicial", type=float, default=LR_INICIAL)
    parser.add_argument("--lr_fine", type=float, default=LR_FINE)
    parser.add_argument("--unfreeze_last_n", type=int, default=UNFREEZE_LAST_N)
    parser.add_argument("--skip_split", action="store_true",
                        help="Set if dataset is already structured into train/val/test folders")
    return parser.parse_args()


def entrenar_pipeline(args):
    seed_everything(RANDOM_STATE)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.results_dir) / f"run_{timestamp}_efficientnetb0"
    (run_dir / "modelo").mkdir(parents=True, exist_ok=True)
    (run_dir / "metricas").mkdir(parents=True, exist_ok=True)
    (run_dir / "graficos").mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data_dir)

    if args.skip_split:
        train_dir = data_path / "train"
        val_dir = data_path / "val"
        test_dir = data_path / "test"
    else:
        print(f"\n[1/4] Preparing stratified dataset splits from: {data_path}")
        dataset_target_dir = run_dir / "dataset"
        train_dir, val_dir, test_dir, stats = preparar_dataset_splits(
            data_path, dataset_target_dir, seed=RANDOM_STATE
        )
        with open(run_dir / "metricas/dataset_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

    print("\n[2/4] Setting up Data Generators & Tissue-Aware Preprocessing")
    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_train_morph,
        rotation_range=20,
        width_shift_range=0.10,
        height_shift_range=0.10,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest"
    )

    val_gen = ImageDataGenerator(preprocessing_function=preprocess_eval_morph)
    test_gen = ImageDataGenerator(preprocessing_function=preprocess_eval_morph)

    train_flow = train_gen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=args.batch_size,
        class_mode="categorical",
        classes=CLASES,
        shuffle=True,
        seed=RANDOM_STATE
    )

    val_flow = val_gen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=args.batch_size,
        class_mode="categorical",
        classes=CLASES,
        shuffle=False
    )

    test_flow = test_gen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=args.batch_size,
        class_mode="categorical",
        classes=CLASES,
        shuffle=False
    )

    print("\n[3/4] Building EfficientNetB0 Transfer Learning Architecture")
    model, base_model = crear_modelo_sequential(args.lr_inicial)

    best_weights_path = run_dir / "modelo/best_weights.weights.h5"
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        keras.callbacks.ModelCheckpoint(filepath=str(best_weights_path), monitor="val_loss",
                                       save_best_only=True, save_weights_only=True, verbose=1),
    ]

    print("\n>>> FASE 1: Training Classification Head (Backbone Frozen)")
    h1 = model.fit(
        train_flow,
        validation_data=val_flow,
        epochs=args.epochs_fase1,
        callbacks=callbacks,
        class_weight=CLASS_WEIGHT,
        verbose=1
    )

    print(f"\n>>> FASE 2: Fine-Tuning Last {args.unfreeze_last_n} Layers")
    base_model.trainable = True
    for layer in base_model.layers[:-args.unfreeze_last_n]:
        layer.trainable = False

    loss = keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)
    model.compile(
        optimizer=keras.optimizers.Adam(args.lr_fine),
        loss=loss,
        metrics=["accuracy"]
    )

    h2 = model.fit(
        train_flow,
        validation_data=val_flow,
        epochs=args.epochs_fase2,
        callbacks=callbacks,
        class_weight=CLASS_WEIGHT,
        verbose=1
    )

    history = {
        "accuracy": h1.history.get("accuracy", []) + h2.history.get("accuracy", []),
        "val_accuracy": h1.history.get("val_accuracy", []) + h2.history.get("val_accuracy", []),
        "loss": h1.history.get("loss", []) + h2.history.get("loss", []),
        "val_loss": h1.history.get("val_loss", []) + h2.history.get("val_loss", []),
    }

    with open(run_dir / "metricas/history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Save weights in pickle format
    pkl_path = run_dir / "modelo/modelo_pesos.pkl"
    guardar_modelo_pickle(model, pkl_path)

    print("\n[4/4] Evaluating Model on Independent Test Set")
    preds = model.predict(test_flow, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_flow.classes

    acc = float(accuracy_score(y_true, y_pred))
    rec = recall_score(y_true, y_pred, average=None)
    prec = precision_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    resultados = {
        "accuracy_global": acc,
        "clases": CLASES,
        "sensibilidad_por_clase": {c: float(rec[i]) for i, c in enumerate(CLASES)},
        "precision_por_clase": {c: float(prec[i]) for i, c in enumerate(CLASES)},
        "f1_por_clase": {c: float(f1[i]) for i, c in enumerate(CLASES)},
        "matriz_confusion": cm.tolist(),
        "matriz_confusion_normalizada": cm_norm.tolist()
    }

    with open(run_dir / "metricas/resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2)

    # Plot Confusion Matrix
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=CLASES, yticklabels=CLASES)
    plt.title("Matriz de Confusión Normalizada (Test Set)")
    plt.xlabel("Predicción")
    plt.ylabel("Verdadero")
    plt.tight_layout()
    plt.savefig(run_dir / "graficos/matriz_confusion_normalizada.png", dpi=300)
    plt.close()

    # Plot Training Curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["accuracy"], label="Train Acc")
    plt.plot(history["val_accuracy"], label="Val Acc")
    plt.title("Evolución de Exactitud")
    plt.xlabel("Época")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Evolución de Pérdida")
    plt.xlabel("Época")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "graficos/curvas_entrenamiento.png", dpi=300)
    plt.close()

    print("\n" + "=" * 60)
    print("RESUMEN DE EVALUACIÓN FINAL:")
    print(f"Accuracy Global: {acc * 100:.2f}%")
    for i, c in enumerate(CLASES):
        print(f"  - {c}: Sensibilidad={rec[i]*100:.2f}%, Precisión={prec[i]*100:.2f}%, F1={f1[i]*100:.2f}%")
    print(f"Resultados guardados en: {run_dir}")
    print("=" * 60)

    return run_dir


if __name__ == "__main__":
    args = parse_args()
    entrenar_pipeline(args)
