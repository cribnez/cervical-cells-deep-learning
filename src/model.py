"""
Model Architecture, Weights Serialization & Explainable AI (Grad-CAM).
"""
import pickle
import json
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import cv2
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0

from .config import (
    IMG_SIZE,
    CLASES,
    LR_INICIAL,
    LABEL_SMOOTHING,
    GRADCAM_LAYER_NAME,
)


def crear_modelo_sequential(lr: float = LR_INICIAL) -> Tuple[keras.Model, keras.Model]:
    """
    Constructs the Sequential model for training phases.
    Returns: (compiled_model, base_model)
    """
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
        pooling="avg"
    )
    base_model.trainable = False

    model = keras.Sequential([
        base_model,
        layers.Dense(256, activation="relu", name="dense_256"),
        layers.Dropout(0.4, name="dropout"),
        layers.Dense(len(CLASES), activation="softmax", name="predictions")
    ], name="efficientnetb0_cervical_seq")

    loss = keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=loss,
        metrics=["accuracy"]
    )
    return model, base_model


def crear_modelo_functional() -> Tuple[keras.Model, keras.Model]:
    """
    Constructs the Functional model needed for Grad-CAM intermediate feature map extraction.
    Returns: (model, base_model)
    """
    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="input_1")
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        pooling=None
    )

    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    outputs = layers.Dense(len(CLASES), activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="efficientnetb0_cervical_func")
    return model, base_model


def guardar_modelo_pickle(model: keras.Model, output_path: Path):
    """
    Serializes layer weights into a lightweight pickle file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pesos = []
    for layer in model.layers:
        pesos.append([w.numpy() for w in layer.weights])

    with open(output_path, "wb") as f:
        pickle.dump(pesos, f)

    config_path = output_path.parent / "config.json"
    config = {
        "clases": CLASES,
        "img_size": list(IMG_SIZE),
        "arquitectura": "EfficientNetB0",
        "n_classes": len(CLASES)
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"✓ Model weights successfully serialized to: {output_path}")


def cargar_pesos_pickle(model: keras.Model, base_model: keras.Model, ruta_pesos: Path) -> keras.Model:
    """
    Loads weights from pickle into the Functional model.
    """
    ruta_pesos = Path(ruta_pesos)
    if not ruta_pesos.exists():
        raise FileNotFoundError(f"Pesos file not found at: {ruta_pesos}")

    with open(ruta_pesos, "rb") as f:
        pesos = pickle.load(f)

    base_model.set_weights(pesos[0])
    model.get_layer("dense_256").set_weights(pesos[1])

    if len(pesos) >= 4 and len(pesos[3]) > 0:
        model.get_layer("predictions").set_weights(pesos[3])
    else:
        last = None
        for blk in reversed(pesos):
            if isinstance(blk, list) and len(blk) > 0:
                last = blk
                break
        if last is None:
            raise ValueError("No matching prediction layer weights found in pickle file.")
        model.get_layer("predictions").set_weights(last)

    # Warmup dummy pass
    _ = model(tf.zeros((1, *IMG_SIZE, 3), dtype=tf.float32), training=False)
    return model


def generar_gradcam_heatmap(
    model: keras.Model,
    base_model: keras.Model,
    x_tensor: np.ndarray,
    class_index: int,
    conv_layer_name: str = GRADCAM_LAYER_NAME
) -> np.ndarray:
    """
    Computes Grad-CAM activation heatmap for the designated target class.
    """
    try:
        conv_layer = base_model.get_layer(conv_layer_name)
    except Exception:
        conv_layer = None
        for layer in reversed(base_model.layers):
            try:
                if len(layer.output.shape) == 4:
                    conv_layer = layer
                    break
            except Exception:
                pass
        if conv_layer is None:
            raise ValueError("No valid 4D convolutional layer found in base model for Grad-CAM.")

    grad_model = keras.Model(inputs=model.inputs, outputs=[conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x_tensor, training=False)
        score = preds[:, class_index]

    grads = tape.gradient(score, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-6)
    return heatmap.numpy()


def generar_gradcam_superpuesto_x(
    model: keras.Model,
    base_model: keras.Model,
    x_model: np.ndarray,
    img_preproc_pil: Image.Image,
    class_index: int,
    conv_layer_name: str = GRADCAM_LAYER_NAME,
    alpha: float = 0.55
) -> Image.Image:
    """
    Overlays Grad-CAM heatmap on top of the preprocessed image that entered the model.
    """
    img_np = np.array(img_preproc_pil, dtype=np.uint8)
    heatmap = generar_gradcam_heatmap(model, base_model, x_model, class_index, conv_layer_name)

    heatmap = cv2.resize(heatmap, IMG_SIZE)
    heatmap_uint8 = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = (alpha * img_np + (1.0 - alpha) * heatmap_color).astype(np.uint8)
    return Image.fromarray(overlay)
