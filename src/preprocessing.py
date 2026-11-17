"""
Preprocessing pipeline: Reinhard Color Normalization, Tissue-Aware Cropping & Augmentation.
"""
import random
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

from .config import (
    IMG_SIZE,
    TARGET_MEANS,
    TARGET_STDS,
    CHROMA_THR_LOOSE,
    L_DARK_THR_LOOSE,
    MIN_TISSUE_AREA,
    MIN_TISSUE_FRAC,
    MIN_CROP_SIDE,
    PAD_EXTRA_FRAC,
    CROP_ZOOM_MIN,
    CROP_ZOOM_MAX,
    P_REINHARD_TRAIN,
    P_GRAYSCALE,
    HUE_DEG,
    SAT_RANGE,
    VAL_RANGE,
    GAMMA_RANGE,
)


def reinhard_normalize_bgr(src_bgr: np.ndarray, target_means=TARGET_MEANS, target_stds=TARGET_STDS) -> np.ndarray:
    """
    Standardizes image color in CIELAB space to align with reference cytology staining.
    """
    src = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(src)

    means = (np.mean(L), np.mean(A), np.mean(B))
    stds = (np.std(L) + 1e-6, np.std(A) + 1e-6, np.std(B) + 1e-6)

    L = (L - means[0]) / stds[0] * target_stds[0] + target_means[0]
    A = (A - means[1]) / stds[1] * target_stds[1] + target_means[1]
    B = (B - means[2]) / stds[2] * target_stds[2] + target_means[2]

    out = cv2.merge([L, A, B])
    out = np.clip(out, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)
    return out


def tissue_mask_lab_loose(rgb: np.ndarray) -> np.ndarray:
    """
    Generates a tissue mask in LAB space ensuring faint cytoplasm is included.
    Filters small isolated artifacts based on minimum connected component area.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.int16)
    L = lab[:, :, 0]
    a = lab[:, :, 1] - 128
    b = lab[:, :, 2] - 128
    chroma = np.sqrt(a * a + b * b)

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


def crop_tissue_aware_rgb_train(rgb: np.ndarray) -> np.ndarray:
    """
    Train mode cropping: extracts bounding box around major cellular cluster
    with generous margin and controlled random zoom.
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
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + ww + pad)
    y2 = min(h, y + hh + pad)

    roi = rgb[y1:y2, x1:x2]
    roi_m = m[y1:y2, x1:x2]

    if min(roi.shape[0], roi.shape[1]) < MIN_CROP_SIDE:
        return rgb

    tissue_frac = float(roi_m.mean())
    if tissue_frac < MIN_TISSUE_FRAC:
        return rgb

    rh, rw = roi.shape[:2]
    scale = random.uniform(CROP_ZOOM_MIN, CROP_ZOOM_MAX)
    ch = max(MIN_CROP_SIDE, int(rh * scale))
    cw = max(MIN_CROP_SIDE, int(rw * scale))

    cy = random.randint(ch // 2, max(ch // 2, rh - ch // 2))
    cx = random.randint(cw // 2, max(cw // 2, rw - cw // 2))

    y1z = max(0, cy - ch // 2)
    y2z = min(rh, cy + ch // 2)
    x1z = max(0, cx - cw // 2)
    x2z = min(rw, cx + cw // 2)

    return roi[y1z:y2z, x1z:x2z]


def crop_tissue_aware_rgb_eval(rgb: np.ndarray) -> np.ndarray:
    """
    Evaluation / Inference mode cropping: deterministic bounding box around major cell cluster.
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
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + ww + pad)
    y2 = min(h, y + hh + pad)

    roi = rgb[y1:y2, x1:x2]
    roi_m = m[y1:y2, x1:x2]

    if min(roi.shape[0], roi.shape[1]) < MIN_CROP_SIDE:
        return rgb

    tissue_frac = float(roi_m.mean())
    if tissue_frac < MIN_TISSUE_FRAC:
        return rgb

    return roi


def hsv_jitter_rgb(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + random.uniform(-HUE_DEG, HUE_DEG)) % 180.0
    hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(*SAT_RANGE), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * random.uniform(*VAL_RANGE), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def random_gamma_rgb(rgb: np.ndarray) -> np.ndarray:
    g = random.uniform(*GAMMA_RANGE)
    x = rgb.astype(np.float32) / 255.0
    x = np.power(x, g)
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def preprocess_train_morph(img: np.ndarray) -> np.ndarray:
    """
    Full training preprocessing pipeline with augmentations.
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


def preprocess_eval_morph(img: np.ndarray) -> np.ndarray:
    """
    Deterministic evaluation/validation preprocessing pipeline.
    """
    x = np.clip(img, 0, 255).astype(np.uint8)

    bgr = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    bgr = reinhard_normalize_bgr(bgr)
    x = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    x = crop_tissue_aware_rgb_eval(x)
    x = cv2.resize(x, IMG_SIZE, interpolation=cv2.INTER_AREA)

    return preprocess_input(x.astype(np.float32))


def preprocess_single_image(image_input):
    """
    Prepares a single image for inference and returns:
    - x_tensor: (1, 224, 224, 3) float32 for EfficientNet
    - img_preproc_pil: PIL image (224x224) showing what the model actually receives
    - img_original_pil: PIL image of original input
    """
    if isinstance(image_input, (str, bytes)):
        img_original_pil = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img_original_pil = image_input.convert("RGB")
    else:
        img_original_pil = Image.fromarray(image_input).convert("RGB")

    rgb = np.array(img_original_pil, dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Reinhard color normalization
    bgr = reinhard_normalize_bgr(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Deterministic secure tissue crop
    rgb = crop_tissue_aware_rgb_eval(rgb)

    # Resize to model dimension
    rgb_224 = cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)
    img_preproc_pil = Image.fromarray(rgb_224)

    # Preprocess tensor
    x = rgb_224.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    x_tensor = preprocess_input(x)

    return x_tensor, img_preproc_pil, img_original_pil
