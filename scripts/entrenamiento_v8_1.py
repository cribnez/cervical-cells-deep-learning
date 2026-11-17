"""
ENTRENAMIENTO IA – CITOLOGÍA CERVICAL v8.1 (MORFOLOGÍA/TRIAJE) + RECORTE SEGURO
- EfficientNetB0 + Transfer Learning
- Enfocado a morfología: Reinhard + recorte tissue-aware (seguro) + augment suave
- EarlyStopping + ReduceLROnPlateau + ModelCheckpoint
- Class weights manual (triaje: prioriza HSIL)
- Guarda: dataset split, history, métricas, matriz de confusión, gráficas, pesos .pkl
"""

# ======================================================
# BLOQUE 0: IMPORTS Y DEPENDENCIAS
# ======================================================
# Este script usa:
# - OpenCV/Numpy para preprocesamiento (color + recorte)
# - Keras/TensorFlow para el modelo y entrenamiento
# - sklearn para split y métricas
# - matplotlib/seaborn para gráficas y matrices de confusión
#
# La idea general del archivo:
# 1) Preparar dataset con split reproducible
# 2) Definir preprocesamiento (train con augment; eval determinístico)
# 3) Entrenar EfficientNetB0 en 2 fases (cabeza y fine-tuning)
# 4) Evaluar en test y guardar métricas/gráficas/pesos
# ======================================================

import os
import shutil
import pickle
import random
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ======================================================
# BLOQUE 1: CONFIGURACIÓN DEL EXPERIMENTO
# ======================================================
# Aquí se define todo lo que controla el entrenamiento:
# - Rutas del dataset original y carpeta de salidas
# - Clases y tamaño de entrada del modelo
# - Hiperparámetros de entrenamiento (épocas / learning rates)
# - Split train/val/test
# - Estrategia de fine-tuning (cuántas capas descongelar)
# - Pesos de clase (enfoque triaje: HSIL pesa más)
# - Parámetros del preprocesamiento y recorte seguro
#
# IMPORTANTE:
# CLASES tiene orden fijo, y ese orden se respeta en:
# - flow_from_directory(classes=CLASES)
# - métricas de triaje (HSIL=0, LSIL=1, NILM=2)
# ======================================================

DATASET_ORIGINAL = r"C:\Users\mbada\Desktop\Proyecto citologia\Imagenes"
RESULTADOS_BASE  = r"C:\Users\mbada\Desktop\Proyecto citologia\Resultados_Entrenamiento"

CLASES = ["HSIL", "LSIL", "NILM"]  # orden fijo
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Entrenamiento
EPOCHS_FASE1 = 20
EPOCHS_FASE2 = 20
LR_INICIAL = 1e-3
LR_FINE = 3e-5

# Split
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_STATE = 42

# Fine-tuning
UNFREEZE_LAST_N = 10  # descongela menos (mejor morfología)

# Loss smoothing (reduce overconfidence)
LABEL_SMOOTHING = 0.03

# Triaje: pesos manuales (HSIL importante)
# (HSIL=0, LSIL=1, NILM=2)
CLASS_WEIGHT = {0: 2.0, 1: 1.2, 2: 0.8}

# Augment “anti-sesgo” SUAVE (no te rompas aprendizaje fino)
P_REINHARD_TRAIN = 0.70
P_GRAYSCALE = 0.15

HUE_DEG = 8
SAT_RANGE = (0.75, 1.35)
VAL_RANGE = (0.75, 1.25)
GAMMA_RANGE = (0.8, 1.3)

# ---------------- RECORTE SEGURO (ANTI NÚCLEO SUELTO) ----------------
# Este recorte está pensado para evitar que el modelo se "vaya" con:
# - puntitos / núcleos aislados / ruido
# y que en cambio se quede con una región que tenga:
# - suficiente tejido/citoplasma alrededor
#
# La máscara es "suave" para incluir citoplasma tenue (no sólo lo oscuro).
# Luego se filtran componentes por área y se recorta alrededor del componente mayor.
# ---------------------------------------------------------------

CHROMA_THR_LOOSE = 6
L_DARK_THR_LOOSE = 245

MIN_TISSUE_AREA = 2500
MIN_TISSUE_FRAC = 0.08
MIN_CROP_SIDE = 160
PAD_EXTRA_FRAC = 0.15

CROP_ZOOM_MIN = 0.80
CROP_ZOOM_MAX = 1.00

# ======================================================


# ======================================================
# BLOQUE 2: REPRODUCIBILIDAD
# ======================================================
# Para que el split y el entrenamiento sean lo más reproducibles posible:
# - seed de random
# - seed de numpy
# - seed de tensorflow
# ======================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ======================================================
# BLOQUE 3: ORGANIZACIÓN DE SALIDAS (RUN)
# ======================================================
# Cada entrenamiento genera una carpeta nueva con timestamp.
# Dentro se guarda:
# - dataset ya dividido (train/val/test)
# - pesos del modelo y config
# - métricas en JSON/TXT
# - gráficas (curvas + matrices)
# ======================================================

def crear_carpeta_resultados():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(RESULTADOS_BASE) / f"run_{timestamp}_v81_morph_safe"

    (base / "dataset/train").mkdir(parents=True, exist_ok=True)
    (base / "dataset/val").mkdir(parents=True, exist_ok=True)
    (base / "dataset/test").mkdir(parents=True, exist_ok=True)
    (base / "modelo").mkdir(parents=True, exist_ok=True)
    (base / "metricas").mkdir(parents=True, exist_ok=True)
    (base / "graficos").mkdir(parents=True, exist_ok=True)

    return base


# ======================================================
# BLOQUE 4: PREPARACIÓN DEL DATASET (SPLIT POR IMAGEN)
# ======================================================
# - Lee las imágenes por clase desde DATASET_ORIGINAL/CLASE
# - Genera split reproducible train/val/test por lista de archivos
# - Copia físicamente las imágenes a la estructura:
#   resultados/dataset/{train,val,test}/{HSIL,LSIL,NILM}
#
# Se guarda dataset_stats.json para documentar cuántas imágenes quedaron en cada split.
# ======================================================

def preparar_dataset(carpeta_resultados):
    print("=" * 60)
    print("PREPARANDO DATASET (split por imagen)")
    print("=" * 60)

    stats = {}

    for clase in CLASES:
        origen = Path(DATASET_ORIGINAL) / clase

        imagenes = (
            list(origen.glob("*.jpg")) +
            list(origen.glob("*.jpeg")) +
            list(origen.glob("*.png")) +
            list(origen.glob("*.JPG")) +
            list(origen.glob("*.PNG")) +
            list(origen.glob("*.tif")) +
            list(origen.glob("*.tiff"))
        )

        if len(imagenes) == 0:
            print(f"[ERROR] No se encontraron imágenes en {origen}")
            continue

        train_val, test = train_test_split(
            imagenes,
            test_size=TEST_SPLIT,
            random_state=RANDOM_STATE,
            shuffle=True
        )

        train, val = train_test_split(
            train_val,
            test_size=VALIDATION_SPLIT / (1 - TEST_SPLIT),
            random_state=RANDOM_STATE
        )

        for split, imgs in zip(["train", "val", "test"], [train, val, test]):
            destino = carpeta_resultados / "dataset" / split / clase
            destino.mkdir(parents=True, exist_ok=True)
            for img in imgs:
                shutil.copy2(img, destino / img.name)

        stats[clase] = {"total": len(imagenes), "train": len(train), "val": len(val), "test": len(test)}
        print(f"{clase}: {stats[clase]}")

    with open(carpeta_resultados / "metricas/dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return (
        carpeta_resultados / "dataset/train",
        carpeta_resultados / "dataset/val",
        carpeta_resultados / "dataset/test"
    )


# ======================================================
# BLOQUE 5: NORMALIZACIÓN DE COLOR (REINHARD EN LAB)
# ======================================================
# Se usa para reducir la variabilidad de color/iluminación entre imágenes.
# La implementación:
# - Convierte BGR -> LAB
# - Estandariza cada canal (L,A,B) a medias/desv objetivo
# - Regresa imagen BGR normalizada
#
# En TRAIN se aplica con probabilidad (P_REINHARD_TRAIN).
# En VAL/TEST se aplica siempre para que el input sea consistente.
# ======================================================

TARGET_MEANS = (145, 128, 128)
TARGET_STDS  = (20, 12, 12)

def reinhard_normalize_bgr(src_bgr, target_means=TARGET_MEANS, target_stds=TARGET_STDS):
    src = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(src)

    means = (np.mean(L), np.mean(A), np.mean(B))
    stds  = (np.std(L) + 1e-6, np.std(A) + 1e-6, np.std(B) + 1e-6)

    L = (L - means[0]) / stds[0] * target_stds[0] + target_means[0]
    A = (A - means[1]) / stds[1] * target_stds[1] + target_means[1]
    B = (B - means[2]) / stds[2] * target_stds[2] + target_means[2]

    out = cv2.merge([L, A, B])
    out = np.clip(out, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)
    return out


# ======================================================
# BLOQUE 6: MÁSCARA DE TEJIDO Y RECORTE SEGURO (TISSUE-AWARE)
# ======================================================
# Objetivo del recorte:
# - Quitar fondo (espacios claros)
# - Evitar "núcleo suelto" o puntos pequeños
# - Mantener región con citoplasma/tejido suficiente
#
# Estrategia:
# 1) Crear máscara "suave" en LAB:
#    - usa chroma y L para detectar material teñido o más oscuro
# 2) Limpiar máscara (median blur, open, close)
# 3) Conectar componentes y filtrar por MIN_TISSUE_AREA
# 4) Recortar alrededor del componente más grande con padding extra
#
# Train: recorte + zoom moderado (aleatorio dentro del ROI)
# Eval: recorte determinístico (sin random)
# ======================================================

def tissue_mask_lab_loose(rgb):
    """
    Máscara de tejido "suave" para incluir citoplasma pálido.
    Elimina puntitos por área mínima.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.int16)
    L = lab[:, :, 0]
    a = lab[:, :, 1] - 128
    b = lab[:, :, 2] - 128
    chroma = np.sqrt(a*a + b*b)

    m = (chroma > CHROMA_THR_LOOSE) | (L < L_DARK_THR_LOOSE)
    m = (m.astype(np.uint8) * 255)

    m = cv2.medianBlur(m, 5)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    out = np.zeros_like(m, dtype=np.uint8)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= MIN_TISSUE_AREA:
            out[labels == i] = 255

    return (out > 0)


def crop_tissue_aware_rgb_train(rgb):
    """
    TRAIN: recorta alrededor del componente más grande (evita núcleo suelto) + zoom moderado.
    Si no cumple seguridad => regresa imagen original.
    """
    h, w = rgb.shape[:2]
    m = tissue_mask_lab_loose(rgb)

    # Si casi no hay tejido, mejor no recortar (evito recortes basura)
    if int(m.sum()) < MIN_TISSUE_AREA:
        return rgb

    num, labels, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if num <= 1:
        return rgb

    # Me quedo con el componente más grande (reduce riesgo de agarrar ruido)
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))  # mayor

    x = stats[idx, cv2.CC_STAT_LEFT]
    y = stats[idx, cv2.CC_STAT_TOP]
    ww = stats[idx, cv2.CC_STAT_WIDTH]
    hh = stats[idx, cv2.CC_STAT_HEIGHT]

    # Padding extra para incluir citoplasma alrededor
    pad = int(PAD_EXTRA_FRAC * max(ww, hh))
    x1 = max(0, x - pad); y1 = max(0, y - pad)
    x2 = min(w, x + ww + pad); y2 = min(h, y + hh + pad)

    roi = rgb[y1:y2, x1:x2]
    roi_m = m[y1:y2, x1:x2]

    # Validaciones de seguridad del recorte (tamaño mínimo y % tejido)
    if min(roi.shape[0], roi.shape[1]) < MIN_CROP_SIDE:
        return rgb

    tissue_frac = float(roi_m.mean())
    if tissue_frac < MIN_TISSUE_FRAC:
        return rgb

    # Zoom aleatorio moderado dentro del ROI para variar encuadre sin perder contexto
    rh, rw = roi.shape[:2]
    scale = random.uniform(CROP_ZOOM_MIN, CROP_ZOOM_MAX)
    ch = max(MIN_CROP_SIDE, int(rh * scale))
    cw = max(MIN_CROP_SIDE, int(rw * scale))

    cy = random.randint(ch // 2, max(ch // 2, rh - ch // 2))
    cx = random.randint(cw // 2, max(cw // 2, rw - cw // 2))

    y1z = max(0, cy - ch // 2); y2z = min(rh, cy + ch // 2)
    x1z = max(0, cx - cw // 2); x2z = min(rw, cx + cw // 2)

    out = roi[y1z:y2z, x1z:x2z]
    return out


def crop_tissue_aware_rgb_eval(rgb):
    """
    EVAL/TEST: determinístico (sin random). Solo bbox del componente mayor + padding.
    Si no cumple seguridad => regresa original.
    """
    h, w = rgb.shape[:2]
    m = tissue_mask_lab_loose(rgb)

    if int(m.sum()) < MIN_TISSUE_AREA:
        return rgb

    num, labels, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if num <= 1:
        return rgb

    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))

    x = stats[idx, cv2.CC_STAT_LEFT]
    y = stats[idx, cv2.CC_STAT_TOP]
    ww = stats[idx, cv2.CC_STAT_WIDTH]
    hh = stats[idx, cv2.CC_STAT_HEIGHT]

    pad = int(PAD_EXTRA_FRAC * max(ww, hh))
    x1 = max(0, x - pad); y1 = max(0, y - pad)
    x2 = min(w, x + ww + pad); y2 = min(h, y + hh + pad)

    roi = rgb[y1:y2, x1:x2]
    roi_m = m[y1:y2, x1:x2]

    if min(roi.shape[0], roi.shape[1]) < MIN_CROP_SIDE:
        return rgb

    tissue_frac = float(roi_m.mean())
    if tissue_frac < MIN_TISSUE_FRAC:
        return rgb

    return roi


# ======================================================
# BLOQUE 7: AUGMENT DE COLOR SUAVE (HSV + GAMMA)
# ======================================================
# Aquí meto augmentación "controlada" para no destruir morfología.
# - jitter en HSV con rangos pequeños
# - corrección gamma aleatoria
#
# Se complementa con augment geométrico del ImageDataGenerator (rotaciones, shifts, etc.).
# ======================================================

def hsv_jitter_rgb(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + random.uniform(-HUE_DEG, HUE_DEG)) % 180.0
    hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(*SAT_RANGE), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * random.uniform(*VAL_RANGE), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

def random_gamma_rgb(rgb):
    g = random.uniform(*GAMMA_RANGE)
    x = rgb.astype(np.float32) / 255.0
    x = np.power(x, g)
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


# ======================================================
# BLOQUE 8: PREPROCESAMIENTO FINAL PARA TRAIN VS EVAL
# ======================================================
# Dos pipelines separados para evitar "data leakage" o inconsistencia:
#
# TRAIN:
# - Reinhard con probabilidad (para no hacer todo idéntico)
# - Recorte tissue-aware con random zoom moderado
# - Resize 224
# - Color jitter + gamma
# - Grayscale con prob baja (regularización)
# - preprocess_input (EfficientNet)
#
# VAL/TEST:
# - Reinhard siempre
# - Recorte tissue-aware determinístico
# - Resize 224
# - preprocess_input
# ======================================================

def preprocess_train_morph(img):
    """
    Train:
    - Reinhard (prob)
    - crop tissue-aware seguro (evita núcleo suelto)
    - resize
    - jitter suave + gamma
    - grayscale (bajo)
    - preprocess_input
    """
    x = np.clip(img, 0, 255).astype(np.uint8)

    if random.random() < P_REINHARD_TRAIN:
        bgr = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        bgr = reinhard_normalize_bgr(bgr)
        x = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    x = crop_tissue_aware_rgb_train(x)
    x = cv2.resize(x, IMG_SIZE, interpolation=cv2.INTER_AREA)

    x = hsv_jitter_rgb(x)
    x = random_gamma_rgb(x)

    if random.random() < P_GRAYSCALE:
        g = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        x = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)

    return preprocess_input(x.astype(np.float32))


def preprocess_eval_morph(img):
    """
    Val/Test (determinístico):
    - Reinhard siempre
    - crop tissue-aware seguro determinístico
    - resize
    - preprocess_input
    """
    x = np.clip(img, 0, 255).astype(np.uint8)

    bgr = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    bgr = reinhard_normalize_bgr(bgr)
    x = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    x = crop_tissue_aware_rgb_eval(x)
    x = cv2.resize(x, IMG_SIZE, interpolation=cv2.INTER_AREA)

    return preprocess_input(x.astype(np.float32))


# ======================================================
# BLOQUE 9: CONSTRUCCIÓN DEL MODELO (TRANSFER LEARNING)
# ======================================================
# - Se carga EfficientNetB0 preentrenada (ImageNet) sin top
# - Se congela el backbone al inicio
# - Se añade una "cabeza" simple:
#   Dense(256) + Dropout + Softmax(3)
#
# Label smoothing ayuda a que el modelo no se vuelva demasiado confiado
# (útil si el dataset tiene ruido de etiquetas).
# ======================================================

def crear_modelo():
    print("\n" + "=" * 60)
    print("CREANDO MODELO – EfficientNetB0 (v8.1 morph safe)")
    print("=" * 60)

    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
        pooling="avg"
    )
    base_model.trainable = False

    model = keras.Sequential([
        base_model,
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(len(CLASES), activation="softmax")
    ])

    loss = keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)

    model.compile(
        optimizer=keras.optimizers.Adam(LR_INICIAL),
        loss=loss,
        metrics=["accuracy"]
    )

    print(f"Parámetros totales: {model.count_params():,}")
    return model, base_model


# ======================================================
# BLOQUE 10: ENTRENAMIENTO (2 FASES) + CALLBACKS
# ======================================================
# Generadores:
# - train_gen usa preprocess_train_morph + augment geométrico estándar
# - val_gen usa preprocess_eval_morph (determinístico)
#
# Callbacks:
# - EarlyStopping: corta si val_loss no mejora (evita sobreentrenar)
# - ReduceLROnPlateau: baja LR si se estanca
# - ModelCheckpoint: guarda el mejor modelo por val_loss
#
# Entrenamiento por fases:
# Fase 1: backbone congelado, entrena solo la cabeza
# Fase 2: descongela últimas UNFREEZE_LAST_N capas para ajustar detalles finos
#
# class_weight se aplica en ambas fases para sesgar el aprendizaje hacia HSIL.
# ======================================================

def entrenar_modelo(model, base_model, train_dir, val_dir, carpeta_resultados):
    print("\n" + "=" * 60)
    print("ENTRENAMIENTO (v8.1 morph safe / triaje)")
    print("=" * 60)

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

    val_gen = ImageDataGenerator(
        preprocessing_function=preprocess_eval_morph
    )

    train_flow = train_gen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASES,
        shuffle=True,
        seed=RANDOM_STATE
    )

    val_flow = val_gen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASES,
        shuffle=False
    )

    print("class_weight (manual triaje):", CLASS_WEIGHT)

    best_path = carpeta_resultados / "modelo/best_weights.weights.h5"
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        keras.callbacks.ModelCheckpoint(filepath=str(best_path), monitor="val_loss",
                                       save_best_only=True, save_weights_only=True, verbose=1),
    ]

    print("\n>>> FASE 1 – Cabeza (base congelada)")
    h1 = model.fit(
        train_flow,
        validation_data=val_flow,
        epochs=EPOCHS_FASE1,
        callbacks=callbacks,
        class_weight=CLASS_WEIGHT,
        verbose=1,
        workers=1,
        use_multiprocessing=False
    )

    print("\n>>> FASE 2 – Fine tuning (últimas capas)")
    base_model.trainable = True
    for layer in base_model.layers[:-UNFREEZE_LAST_N]:
        layer.trainable = False

    loss = keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)
    model.compile(
        optimizer=keras.optimizers.Adam(LR_FINE),
        loss=loss,
        metrics=["accuracy"]
    )

    h2 = model.fit(
        train_flow,
        validation_data=val_flow,
        epochs=EPOCHS_FASE2,
        callbacks=callbacks,
        class_weight=CLASS_WEIGHT,
        verbose=1,
        workers=1,
        use_multiprocessing=False
    )

    # Unifico history de las 2 fases para graficar todo junto
    history = {
        "accuracy": h1.history["accuracy"] + h2.history["accuracy"],
        "val_accuracy": h1.history["val_accuracy"] + h2.history["val_accuracy"],
        "loss": h1.history["loss"] + h2.history["loss"],
        "val_loss": h1.history["val_loss"] + h2.history["val_loss"],
    }

    # Guardo hiperparámetros y history para dejar trazabilidad del run
    hp = {
        "IMG_SIZE": IMG_SIZE, "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS_FASE1": EPOCHS_FASE1, "EPOCHS_FASE2": EPOCHS_FASE2,
        "LR_INICIAL": LR_INICIAL, "LR_FINE": LR_FINE,
        "LABEL_SMOOTHING": LABEL_SMOOTHING,
        "UNFREEZE_LAST_N": UNFREEZE_LAST_N,
        "P_REINHARD_TRAIN": P_REINHARD_TRAIN, "P_GRAYSCALE": P_GRAYSCALE,
        "HUE_DEG": HUE_DEG, "SAT_RANGE": SAT_RANGE, "VAL_RANGE": VAL_RANGE, "GAMMA_RANGE": GAMMA_RANGE,
        "MIN_TISSUE_AREA": MIN_TISSUE_AREA, "MIN_TISSUE_FRAC": MIN_TISSUE_FRAC, "MIN_CROP_SIDE": MIN_CROP_SIDE,
        "PAD_EXTRA_FRAC": PAD_EXTRA_FRAC, "CROP_ZOOM": [CROP_ZOOM_MIN, CROP_ZOOM_MAX],
        "CLASS_WEIGHT": CLASS_WEIGHT
    }

    with open(carpeta_resultados / "metricas/hyperparams.json", "w", encoding="utf-8") as f:
        json.dump(hp, f, indent=2)

    with open(carpeta_resultados / "metricas/history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return model, history


# ======================================================
# BLOQUE 11: GUARDADO DE PESOS (PICKLE) + CONFIG
# ======================================================
# Aquí guardo los pesos por capas en un .pkl.
# Esto sirve para cargar el modelo en otras interfaces/experimentos
# sin depender de un archivo .h5 completo.
#
# También guardo config.json con clases, tamaño y arquitectura.
# ======================================================

def guardar_modelo_pickle(model, carpeta_resultados):
    print("\nGuardando modelo (pickle)...")
    pesos = []
    for layer in model.layers:
        pesos.append([w.numpy() for w in layer.weights])

    ruta_pesos = carpeta_resultados / "modelo/modelo_pesos.pkl"
    with open(ruta_pesos, "wb") as f:
        pickle.dump(pesos, f)

    print(f"✓ Pesos guardados en: {ruta_pesos}")

    config = {"clases": CLASES, "img_size": IMG_SIZE, "arquitectura": "EfficientNetB0"}
    with open(carpeta_resultados / "modelo/config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


# ======================================================
# BLOQUE 12: EVALUACIÓN EN TEST + MÉTRICAS DE TRIAJE
# ======================================================
# Se corre inferencia en test con preprocess_eval_morph (determinístico).
# Métricas estándar:
# - accuracy, precision/recall/f1 por clase
# - matriz de confusión
#
# Métricas enfocadas a triaje:
# - Sensibilidad HSIL: que NO se vaya a NILM (seguridad)
# - Especificidad NILM: que NILM se quede en NILM (eficiencia)
#
# Luego se guardan resultados.json y resultados.txt para reporte.
# ======================================================

def evaluar_modelo(model, test_dir, carpeta_resultados):
    print("\n" + "=" * 60)
    print("EVALUACIÓN EN TEST SET (preprocess_eval_morph)")
    print("=" * 60)

    test_gen = ImageDataGenerator(preprocessing_function=preprocess_eval_morph)
    test_flow = test_gen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASES,
        shuffle=False
    )

    preds = model.predict(test_flow, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_flow.classes

    acc = accuracy_score(y_true, y_pred)
    recall_per_class = recall_score(y_true, y_pred, average=None)
    precision_per_class = precision_score(y_true, y_pred, average=None)
    f1_per_class = f1_score(y_true, y_pred, average=None)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASES)

    # --- Métricas de TRIAJE ---
    HSIL = 0; LSIL = 1; NILM = 2
    hsil_total = int((y_true == HSIL).sum())
    hsil_as_review = int(((y_true == HSIL) & (y_pred != NILM)).sum())
    triage_sens_hsil = hsil_as_review / (hsil_total + 1e-6)

    nilm_total = int((y_true == NILM).sum())
    nilm_as_neg = int(((y_true == NILM) & (y_pred == NILM)).sum())
    triage_spec_nilm = nilm_as_neg / (nilm_total + 1e-6)

    print(f"\nAccuracy (3 clases): {acc*100:.2f}%")
    print("\nMatriz de confusión:\n", cm)
    print("\nReport:\n", report)
    print(f"\nTRIAJE (Revisión=HSIL/LSIL, Negativo=NILM):")
    print(f"  Sensibilidad HSIL (no mandar HSIL a NILM): {triage_sens_hsil*100:.2f}%")
    print(f"  Especificidad NILM (dejar NILM en NILM):   {triage_spec_nilm*100:.2f}%")

    metricas = {
        "accuracy_3clases": float(acc),
        "recall_por_clase": {CLASES[i]: float(recall_per_class[i]) for i in range(len(CLASES))},
        "precision_por_clase": {CLASES[i]: float(precision_per_class[i]) for i in range(len(CLASES))},
        "f1_por_clase": {CLASES[i]: float(f1_per_class[i]) for i in range(len(CLASES))},
        "matriz_confusion": cm.tolist(),
        "triaje_sens_hsil": float(triage_sens_hsil),
        "triaje_spec_nilm": float(triage_spec_nilm),
    }

    with open(carpeta_resultados / "metricas/resultados.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2)

    with open(carpeta_resultados / "metricas/resultados.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy (3 clases): {acc*100:.2f}%\n\n")
        f.write("Recall por clase:\n")
        for i, c in enumerate(CLASES):
            f.write(f"  {c}: {recall_per_class[i]*100:.2f}%\n")
        f.write("\nTRIAJE (Revisión=HSIL/LSIL, Negativo=NILM):\n")
        f.write(f"  Sensibilidad HSIL: {triage_sens_hsil*100:.2f}%\n")
        f.write(f"  Especificidad NILM: {triage_spec_nilm*100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")

    return acc, cm, recall_per_class


# ======================================================
# BLOQUE 13: GRÁFICAS DE ENTRENAMIENTO Y MATRICES
# ======================================================
# Genero:
# - Curvas accuracy (train/val)
# - Curvas loss (train/val)
# - Matriz de confusión absoluta
# - Matriz de confusión normalizada por fila (recall por clase)
#
# Se guardan en /graficos dentro del run.
# ======================================================

def generar_graficos(history, cm, carpeta_resultados):
    # Accuracy
    plt.figure(figsize=(10, 4))
    plt.plot(history["accuracy"], label="Train Acc")
    plt.plot(history["val_accuracy"], label="Val Acc")
    plt.title("Accuracy")
    plt.xlabel("Época")
    plt.ylabel("Acc")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(carpeta_resultados / "graficos/entrenamiento_accuracy.png", dpi=150)
    plt.close()

    # Loss
    plt.figure(figsize=(10, 4))
    plt.plot(history["loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Loss")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(carpeta_resultados / "graficos/entrenamiento_loss.png", dpi=150)
    plt.close()

    # Confusion
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASES, yticklabels=CLASES)
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(carpeta_resultados / "graficos/matriz_confusion.png", dpi=150)
    plt.close()

    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASES, yticklabels=CLASES)
    plt.title("Matriz de Confusión (Normalizada)")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(carpeta_resultados / "graficos/matriz_confusion_normalizada.png", dpi=150)
    plt.close()

    print("✓ Gráficos guardados")


# ======================================================
# BLOQUE 14: MAIN (ORQUESTACIÓN DEL PIPELINE)
# ======================================================
# main() solo coordina el flujo:
# 1) Semillas
# 2) Crear carpeta run
# 3) Preparar dataset
# 4) Crear modelo
# 5) Entrenar (2 fases)
# 6) Evaluar test
# 7) Generar gráficas
# 8) Guardar pesos/config
#
# Se usa try/except para que si algo falla se vea el traceback
# sin perder el path del run que se creó.
# ======================================================

def main():
    seed_everything(RANDOM_STATE)

    print("\n" + "=" * 60)
    print("  ENTRENAMIENTO IA - CITOLOGÍA CERVICAL v8.1 (MORFOLOGÍA/TRIAJE) + RECORTE SEGURO")
    print("=" * 60)

    carpeta_resultados = crear_carpeta_resultados()
    print(f"\nResultados en: {carpeta_resultados}\n")

    try:
        train_dir, val_dir, test_dir = preparar_dataset(carpeta_resultados)

        model, base_model = crear_modelo()

        model, history = entrenar_modelo(model, base_model, train_dir, val_dir, carpeta_resultados)

        acc, cm, recall_per_class = evaluar_modelo(model, test_dir, carpeta_resultados)

        generar_graficos(history, cm, carpeta_resultados)

        guardar_modelo_pickle(model, carpeta_resultados)

        print("\n" + "=" * 60)
        print("  FINALIZADO")
        print("=" * 60)
        print(f"  Accuracy (3 clases): {acc*100:.2f}%")
        print(f"  Recall HSIL: {recall_per_class[0]*100:.2f}%")
        print(f"  Recall LSIL: {recall_per_class[1]*100:.2f}%")
        print(f"  Recall NILM: {recall_per_class[2]*100:.2f}%")
        print(f"  Carpeta: {carpeta_resultados}")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
