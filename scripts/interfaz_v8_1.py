"""
INTERFAZ - Clasificador Citología Cervical v8.1 + TRIAJE + Grad-CAM (UI PRO)
- 1 imagen (campo)
- predicción HSIL/LSIL/NILM
- muestra ORIGINAL / PREPROCESADA (igual que entrenamiento)
- Grad-CAM superpuesto (sobre la imagen preprocesada)
- UI mejorada: paneles, pestañas, estado del modelo, umbral editable, export overlay
"""

# ======================================================
# BLOQUE 0: IMPORTS
# ======================================================
# - Tkinter/ttk: UI (botones, estilos, progress, pestañas)
# - PIL: mostrar imágenes dentro de Tkinter
# - OpenCV/Numpy: preprocesamiento y Grad-CAM overlay
# - TensorFlow/Keras: cargar modelo y predecir
# - Pickle: cargar pesos guardados del entrenamiento
# ======================================================

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import pickle
from pathlib import Path
import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input


# ======================================================
# BLOQUE 1: CONFIG PRINCIPAL
# ======================================================
# - RUTA_PESOS: apunta al .pkl generado por tu entrenamiento v8.1
# - CLASES: orden fijo (importante para indexación)
# - IMG_SIZE: tamaño esperado por EfficientNetB0 (224x224)
# - COLORES: paleta para UI clínica (rápida de leer)
# - NILM_NEG_THRESHOLD: umbral de triaje (editable en la UI)
# - GRADCAM_LAYER_NAME: capa conv objetivo para Grad-CAM
# ======================================================

RUTA_PESOS = r"C:\Users\mbada\Desktop\Proyecto citologia\Resultados_Entrenamiento\run_20260125_195820_v81_morph_safe\modelo\modelo_pesos.pkl"

CLASES = ["HSIL", "LSIL", "NILM"]
IMG_SIZE = (224, 224)

COLORES = {
    "HSIL": "#e74c3c",
    "LSIL": "#f39c12",
    "NILM": "#27ae60"
}

NILM_NEG_THRESHOLD = 0.90
GRADCAM_LAYER_NAME = "top_conv"


# ======================================================
# BLOQUE 2: PREPROCESAMIENTO v8.1 (IGUAL QUE ENTRENAMIENTO)
# ======================================================
# Este bloque replica el pipeline determinístico usado en VAL/TEST:
# 1) Reinhard (normalización de color)
# 2) Máscara de tejido + recorte seguro determinístico
# 3) Resize a 224
# 4) preprocess_input de EfficientNet
# ======================================================

TARGET_MEANS = (145, 128, 128)
TARGET_STDS  = (20, 12, 12)

CHROMA_THR_LOOSE = 6
L_DARK_THR_LOOSE = 245
MIN_TISSUE_AREA = 2500
MIN_TISSUE_FRAC = 0.08
MIN_CROP_SIDE = 160
PAD_EXTRA_FRAC = 0.15


def reinhard_normalize_bgr(src_bgr, target_means=TARGET_MEANS, target_stds=TARGET_STDS):
    src = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(src)

    means = (np.mean(L), np.mean(A), np.mean(B))
    stds  = (np.std(L)+1e-6, np.std(A)+1e-6, np.std(B)+1e-6)

    L = (L - means[0]) / stds[0] * target_stds[0] + target_means[0]
    A = (A - means[1]) / stds[1] * target_stds[1] + target_means[1]
    B = (B - means[2]) / stds[2] * target_stds[2] + target_means[2]

    out = cv2.merge([L, A, B])
    out = np.clip(out, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)
    return out


def tissue_mask_lab_loose(rgb):
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


def crop_tissue_aware_rgb_eval(rgb):
    h, w = rgb.shape[:2]
    m = tissue_mask_lab_loose(rgb)

    if int(m.sum()) < MIN_TISSUE_AREA:
        return rgb

    num, labels, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if num <= 1:
        return rgb

    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))  # componente mayor

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


def preprocesar_imagen_v81(ruta_imagen):
    """
    Devuelve:
    - x_model: input (1,224,224,3) listo
    - img_preproc_pil: PIL 224x224 que ENTRÓ al modelo (para mostrar/GradCAM)
    - img_original_pil: PIL original
    """
    img_original_pil = Image.open(ruta_imagen).convert("RGB")
    rgb = np.array(img_original_pil, dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Reinhard
    bgr = reinhard_normalize_bgr(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Recorte seguro (determinístico)
    rgb = crop_tissue_aware_rgb_eval(rgb)

    # Resize a 224
    rgb_224 = cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)
    img_preproc_pil = Image.fromarray(rgb_224)

    # Model input
    x = rgb_224.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return x, img_preproc_pil, img_original_pil


# ======================================================
# BLOQUE 3: MODELO (FUNCTIONAL) + CARGA DE PESOS PICKLE
# ======================================================
# Uso modelo Functional porque Grad-CAM necesita acceder
# a tensores intermedios (salida conv).
#
# La carga de pesos asume el formato de tu pickle:
# - pesos[0] -> backbone (EfficientNet)
# - pesos[1] -> dense_256
# - pesos[3] o fallback -> predictions
# ======================================================

def crear_modelo_functional():
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

    model = keras.Model(inputs=inputs, outputs=outputs, name="sovic_model")
    return model, base_model


def cargar_pesos_pickle(model, base_model, ruta_pesos):
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
            raise ValueError("No encontré pesos para la capa de salida en el pickle.")
        model.get_layer("predictions").set_weights(last)

    _ = model(tf.zeros((1, 224, 224, 3), dtype=tf.float32), training=False)
    return model


# ======================================================
# BLOQUE 4: GRAD-CAM
# ======================================================
# - Busca la capa conv objetivo (top_conv por defecto)
# - Calcula gradientes del score de la clase predicha
# - Genera heatmap normalizado
# - Lo superpone sobre la imagen PREPROCESADA (la que entró al modelo)
# ======================================================

def generar_gradcam_superpuesto_x(model, base_model, x_model, img_preproc_pil, class_index,
                                 conv_layer_name=GRADCAM_LAYER_NAME):
    """
    Grad-CAM sobre la imagen PREPROCESADA (la que entra al modelo).
    """
    img_np = np.array(img_preproc_pil, dtype=np.uint8)

    # capa conv
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
            raise ValueError("No se encontró una capa conv válida para Grad-CAM.")

    grad_model = keras.Model(inputs=model.inputs, outputs=[conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x_model, training=False)
        score = preds[:, class_index]

    grads = tape.gradient(score, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-6)
    heatmap = heatmap.numpy()

    heatmap = cv2.resize(heatmap, IMG_SIZE)
    heatmap_uint8 = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = (0.55 * img_np + 0.45 * heatmap_color).astype(np.uint8)
    return Image.fromarray(overlay)


# ======================================================
# BLOQUE 5: UI PRO (APP)
# ======================================================
# Objetivo de esta UI:
# - Se vea clínica/profesional y sea fácil de defender
# - Layout claro:
#   * Izquierda: pestañas de imagen (Original / Preproc / GradCAM)
#   * Derecha: barras de probabilidad + triaje + controles
# - Control de umbral NILM en vivo (slider)
# - Botón para seleccionar imagen y botón para guardar GradCAM
# - Estado del modelo (cargado o error)
# ======================================================

class AplicacionClasificador:
    def __init__(self, root):
        self.root = root
        self.root.title("Clasificador de Citología Cervical v8.1 — Triaje + Grad-CAM")
        self.root.geometry("1050x720")
        self.root.minsize(980, 680)

        # Estado del modelo / datos actuales
        self.model = None
        self.base_model = None

        self.ruta_actual = None
        self.idx_clase_actual = None

        self.img_original_pil = None
        self.img_preproc_pil = None
        self.img_gradcam_pil = None
        self.x_actual = None
        self.prob_actual = None

        # Umbral de triaje (editable)
        self.var_threshold = tk.DoubleVar(value=float(NILM_NEG_THRESHOLD))

        # Configurar tema ttk y estilos
        self._setup_theme()

        # Construir UI
        self._build_layout()

        # Cargar modelo al inicio
        self._cargar_modelo_inicial(RUTA_PESOS)

    # -------------------------
    # Estilo/Tema (look pro)
    # -------------------------
    def _setup_theme(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        self.root.configure(bg="#111827")  # fondo gris oscuro

        style.configure("Card.TFrame", background="#0B1220")
        style.configure("Title.TLabel", background="#111827", foreground="white", font=("Arial", 18, "bold"))
        style.configure("Sub.TLabel", background="#111827", foreground="#9CA3AF", font=("Arial", 10))

        style.configure("PanelTitle.TLabel", background="#0B1220", foreground="white", font=("Arial", 12, "bold"))
        style.configure("Info.TLabel", background="#0B1220", foreground="#CBD5E1", font=("Arial", 10))
        style.configure("Status.TLabel", background="#111827", foreground="#9CA3AF", font=("Arial", 9))

        style.configure("Primary.TButton", font=("Arial", 11, "bold"))
        style.configure("Secondary.TButton", font=("Arial", 10, "bold"))

        # Progresos (uno por clase)
        for c in CLASES:
            style.configure(f"{c}.Horizontal.TProgressbar", troughcolor="#1F2937", background=COLORES[c])

        # Notebook
        style.configure("TNotebook", background="#111827", borderwidth=0)
        style.configure("TNotebook.Tab", padding=(12, 6), font=("Arial", 10, "bold"))

    # -------------------------
    # Layout principal
    # -------------------------
    def _build_layout(self):
        # Header
        header = tk.Frame(self.root, bg="#111827")
        header.pack(fill="x", padx=18, pady=(14, 8))

        ttk.Label(header, text="Clasificador de Citología Cervical", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="v8.1 (morfología) — HSIL / LSIL / NILM | Triaje por umbral NILM | Grad-CAM",
                  style="Sub.TLabel").pack(anchor="w", pady=(2, 0))

        # Body container
        body = tk.Frame(self.root, bg="#111827")
        body.pack(fill="both", expand=True, padx=18, pady=10)

        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=2)
        body.grid_rowconfigure(0, weight=1)

        # Panel izquierda: visor con pestañas
        self.left_card = ttk.Frame(body, style="Card.TFrame")
        self.left_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.left_card.grid_rowconfigure(1, weight=1)
        self.left_card.grid_columnconfigure(0, weight=1)

        ttk.Label(self.left_card, text="Vista de Imagen", style="PanelTitle.TLabel").grid(
            row=0, column=0, sticky="w", padx=14, pady=(12, 6)
        )

        self.notebook = ttk.Notebook(self.left_card)
        self.notebook.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))

        self.tab_original = ttk.Frame(self.notebook)
        self.tab_preproc = ttk.Frame(self.notebook)
        self.tab_gradcam = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_original, text="Original")
        self.notebook.add(self.tab_preproc, text="Preprocesada")
        self.notebook.add(self.tab_gradcam, text="Grad-CAM")

        # Label de imagen por tab
        self.lbl_img_original = tk.Label(self.tab_original, bg="#0B1220", fg="#9CA3AF", text="Sin imagen")
        self.lbl_img_preproc  = tk.Label(self.tab_preproc,  bg="#0B1220", fg="#9CA3AF", text="Sin imagen")
        self.lbl_img_gradcam  = tk.Label(self.tab_gradcam,  bg="#0B1220", fg="#9CA3AF", text="Sin imagen")

        for lbl in (self.lbl_img_original, self.lbl_img_preproc, self.lbl_img_gradcam):
            lbl.pack(expand=True, fill="both", padx=8, pady=8)

        # Panel derecha: resultados + controles
        self.right_card = ttk.Frame(body, style="Card.TFrame")
        self.right_card.grid(row=0, column=1, sticky="nsew")
        self.right_card.grid_columnconfigure(0, weight=1)

        ttk.Label(self.right_card, text="Resultados", style="PanelTitle.TLabel").grid(
            row=0, column=0, sticky="w", padx=14, pady=(12, 6)
        )

        # Estado modelo + ruta pesos
        self.lbl_model_status = ttk.Label(self.right_card, text="Modelo: (cargando...)", style="Info.TLabel")
        self.lbl_model_status.grid(row=1, column=0, sticky="w", padx=14)

        self.lbl_pesos = ttk.Label(self.right_card, text=f"Pesos: {RUTA_PESOS}", style="Info.TLabel", wraplength=360)
        self.lbl_pesos.grid(row=2, column=0, sticky="w", padx=14, pady=(4, 10))

        # Botones principales
        btns = tk.Frame(self.right_card, bg="#0B1220")
        btns.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 10))
        btns.grid_columnconfigure(0, weight=1)
        btns.grid_columnconfigure(1, weight=1)

        self.btn_select_img = ttk.Button(btns, text="📁 Seleccionar imagen", style="Primary.TButton",
                                         command=self.seleccionar_imagen)
        self.btn_select_img.grid(row=0, column=0, sticky="ew", padx=(0, 6), ipady=6)

        self.btn_select_pesos = ttk.Button(btns, text="🧠 Cambiar pesos", style="Secondary.TButton",
                                           command=self.seleccionar_pesos)
        self.btn_select_pesos.grid(row=0, column=1, sticky="ew", padx=(6, 0), ipady=6)

        # Barras de probabilidad
        self.labels_prob = {}
        self.barras_prob = {}

        prob_frame = tk.Frame(self.right_card, bg="#0B1220")
        prob_frame.grid(row=4, column=0, sticky="ew", padx=14, pady=(8, 8))
        prob_frame.grid_columnconfigure(1, weight=1)

        for r, clase in enumerate(CLASES):
            lbl = ttk.Label(prob_frame, text=f"{clase}", style="Info.TLabel")
            lbl.grid(row=r, column=0, sticky="w", pady=6)

            bar = ttk.Progressbar(prob_frame, mode="determinate", length=220,
                                  style=f"{clase}.Horizontal.TProgressbar")
            bar.grid(row=r, column=1, sticky="ew", padx=(10, 10), pady=6)

            val = ttk.Label(prob_frame, text="0.00%", style="Info.TLabel")
            val.grid(row=r, column=2, sticky="e", pady=6)

            self.barras_prob[clase] = bar
            self.labels_prob[clase] = val

        # Predicción principal
        self.lbl_pred = tk.Label(self.right_card, text="Predicción: —", bg="#0B1220",
                                 fg="white", font=("Arial", 16, "bold"))
        self.lbl_pred.grid(row=5, column=0, sticky="ew", padx=14, pady=(10, 4))

        # Triaje
        self.lbl_triaje = tk.Label(self.right_card, text="Triaje: —", bg="#0B1220",
                                   fg="#CBD5E1", font=("Arial", 12, "bold"))
        self.lbl_triaje.grid(row=6, column=0, sticky="ew", padx=14, pady=(0, 10))

        # Slider umbral
        thr_frame = tk.Frame(self.right_card, bg="#0B1220")
        thr_frame.grid(row=7, column=0, sticky="ew", padx=14, pady=(4, 6))
        thr_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(thr_frame, text="Umbral NILM (Negativo):", style="Info.TLabel").grid(row=0, column=0, sticky="w")
        self.lbl_thr_val = ttk.Label(thr_frame, text=f"{self.var_threshold.get():.2f}", style="Info.TLabel")
        self.lbl_thr_val.grid(row=0, column=2, sticky="e")

        self.scale_thr = ttk.Scale(thr_frame, from_=0.50, to=0.99, variable=self.var_threshold,
                                   command=self._on_threshold_change)
        self.scale_thr.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(6, 0))

        # Acciones de imagen (GradCAM / Guardar)
        action_frame = tk.Frame(self.right_card, bg="#0B1220")
        action_frame.grid(row=8, column=0, sticky="ew", padx=14, pady=(10, 10))
        action_frame.grid_columnconfigure(0, weight=1)
        action_frame.grid_columnconfigure(1, weight=1)

        self.btn_make_gradcam = ttk.Button(action_frame, text="🔥 Generar Grad-CAM",
                                           style="Primary.TButton", command=self.generar_gradcam,
                                           state="disabled")
        self.btn_make_gradcam.grid(row=0, column=0, sticky="ew", padx=(0, 6), ipady=5)

        self.btn_save_gradcam = ttk.Button(action_frame, text="💾 Guardar Grad-CAM",
                                           style="Secondary.TButton", command=self.guardar_gradcam,
                                           state="disabled")
        self.btn_save_gradcam.grid(row=0, column=1, sticky="ew", padx=(6, 0), ipady=5)

        # Nota / explicación corta (pro y defendible)
        self.lbl_info = tk.Label(
            self.right_card,
            text=(
                "Entrada del modelo: Reinhard + recorte seguro + resize 224.\n"
                "Regla de triaje: NEGATIVO solo si p(NILM) ≥ umbral; si no, se manda a REVISIÓN.\n"
                "Grad-CAM se calcula sobre la imagen PREPROCESADA (lo que realmente vio el modelo)."
            ),
            bg="#0B1220",
            fg="#9CA3AF",
            font=("Arial", 9),
            justify="left",
            wraplength=420
        )
        self.lbl_info.grid(row=9, column=0, sticky="w", padx=14, pady=(6, 10))

        # Barra de estado inferior
        self.status = ttk.Label(self.root, text="Listo.", style="Status.TLabel")
        self.status.pack(fill="x", padx=18, pady=(0, 10))

    # -------------------------
    # Utilidades UI
    # -------------------------
    def _set_status(self, msg: str):
        self.status.configure(text=msg)
        self.root.update_idletasks()

    def _set_image_on_label(self, lbl: tk.Label, img_pil: Image.Image):
        if img_pil is None:
            lbl.configure(image="", text="Sin imagen")
            lbl.image = None
            return

        # Escalado responsivo (mantiene proporción y se ve pro)
        w = max(520, self.left_card.winfo_width() - 80)
        h = max(520, self.left_card.winfo_height() - 150)

        img_show = img_pil.copy()
        img_show.thumbnail((w, h))
        foto = ImageTk.PhotoImage(img_show)
        lbl.configure(image=foto, text="")
        lbl.image = foto

    def _update_view_tabs(self):
        self._set_image_on_label(self.lbl_img_original, self.img_original_pil)
        self._set_image_on_label(self.lbl_img_preproc, self.img_preproc_pil)
        self._set_image_on_label(self.lbl_img_gradcam, self.img_gradcam_pil)

    # -------------------------
    # Modelo
    # -------------------------
    def _cargar_modelo_inicial(self, ruta_pesos: str):
        try:
            self._set_status("Cargando modelo v8.1...")
            self.model, self.base_model = crear_modelo_functional()
            self.model = cargar_pesos_pickle(self.model, self.base_model, ruta_pesos)
            self.lbl_model_status.configure(text="Modelo: ✓ cargado correctamente")
            self._set_status("Modelo cargado.")
        except Exception as e:
            self.model = None
            self.base_model = None
            self.lbl_model_status.configure(text=f"Modelo: ✗ error al cargar")
            self._set_status(f"Error cargando modelo: {e}")
            messagebox.showerror("Error", f"No se pudo cargar el modelo.\n\nDetalle:\n{e}")

    def seleccionar_pesos(self):
        ruta = filedialog.askopenfilename(
            title="Seleccionar archivo de pesos (.pkl)",
            filetypes=[("Pickle", "*.pkl"), ("Todos", "*.*")]
        )
        if not ruta:
            return

        if not Path(ruta).exists():
            messagebox.showerror("Error", "La ruta seleccionada no existe.")
            return

        self.lbl_pesos.configure(text=f"Pesos: {ruta}")
        self._cargar_modelo_inicial(ruta)

    # -------------------------
    # Umbral triaje en vivo
    # -------------------------
    def _on_threshold_change(self, _evt=None):
        self.lbl_thr_val.configure(text=f"{self.var_threshold.get():.2f}")
        # Si ya hay predicción hecha, recalculo sólo el texto de triaje (sin volver a inferir)
        if self.prob_actual is not None:
            self._actualizar_triaje(self.prob_actual)

    # -------------------------
    # Carga de imagen + predicción
    # -------------------------
    def seleccionar_imagen(self):
        ruta = filedialog.askopenfilename(
            title="Seleccionar imagen de citología (campo)",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"), ("Todos", "*.*")]
        )
        if ruta:
            self.analizar_imagen(ruta)

    def analizar_imagen(self, ruta):
        if self.model is None:
            messagebox.showwarning("Modelo no cargado", "No hay modelo cargado. Revisa pesos.")
            return

        self.ruta_actual = ruta
        self.img_gradcam_pil = None
        self.btn_save_gradcam.config(state="disabled")

        # Preprocesamiento
        try:
            self._set_status("Preprocesando imagen (v8.1)...")
            x, img_preproc_pil, img_original_pil = preprocesar_imagen_v81(ruta)

            self.x_actual = x
            self.img_preproc_pil = img_preproc_pil
            self.img_original_pil = img_original_pil
            self._update_view_tabs()

            # Ir a pestaña Original automáticamente
            self.notebook.select(self.tab_original)

        except Exception as e:
            self._set_status(f"Error preprocesando: {e}")
            messagebox.showerror("Error", f"Error preprocesando la imagen:\n\n{e}")
            return

        # Predicción
        try:
            self._set_status("Ejecutando predicción...")
            prob = self.model.predict(self.x_actual, verbose=0)[0]
            self.prob_actual = prob

            # Actualizar barras + %
            for i, clase in enumerate(CLASES):
                p = float(prob[i]) * 100.0
                self.barras_prob[clase]["value"] = p
                self.labels_prob[clase].configure(text=f"{p:.2f}%")

            idx_max = int(np.argmax(prob))
            clase_predicha = CLASES[idx_max]
            confianza = float(prob[idx_max]) * 100.0
            self.idx_clase_actual = idx_max

            self.lbl_pred.configure(text=f"Predicción: {clase_predicha} ({confianza:.1f}%)",
                                    fg=COLORES[clase_predicha])

            # Triaje
            self._actualizar_triaje(prob)

            # Habilitar Grad-CAM
            self.btn_make_gradcam.config(state="normal")
            self._set_status("Predicción lista. Puedes generar Grad-CAM.")

        except Exception as e:
            self._set_status(f"Error en predicción: {e}")
            messagebox.showerror("Error", f"Error ejecutando predicción:\n\n{e}")

    def _actualizar_triaje(self, prob_vec):
        thr = float(self.var_threshold.get())
        p_nilm = float(prob_vec[2])

        if p_nilm >= thr:
            triage = f"✅ TRIAJE: PROBABLE NEGATIVO  |  p(NILM)={p_nilm:.2f}  (umbral={thr:.2f})"
            color = "#22c55e"
        else:
            triage = f"⚠️ TRIAJE: REVISIÓN         |  p(NILM)={p_nilm:.2f}  (umbral={thr:.2f})"
            color = "#f59e0b"

        self.lbl_triaje.configure(text=triage, fg=color)

    # -------------------------
    # Grad-CAM + Guardar
    # -------------------------
    def generar_gradcam(self):
        if self.model is None or self.base_model is None:
            messagebox.showerror("Error", "Modelo no cargado.")
            return
        if self.x_actual is None or self.idx_clase_actual is None or self.img_preproc_pil is None:
            messagebox.showwarning("Falta información", "Primero carga una imagen y realiza predicción.")
            return

        try:
            self._set_status("Generando Grad-CAM...")
            overlay = generar_gradcam_superpuesto_x(
                self.model, self.base_model,
                self.x_actual, self.img_preproc_pil,
                self.idx_clase_actual,
                conv_layer_name=GRADCAM_LAYER_NAME
            )
            self.img_gradcam_pil = overlay
            self._update_view_tabs()
            self.notebook.select(self.tab_gradcam)

            self.btn_save_gradcam.config(state="normal")
            self._set_status("Grad-CAM listo.")

        except Exception as e:
            self._set_status(f"Grad-CAM error: {e}")
            messagebox.showerror("Error", f"No se pudo generar Grad-CAM:\n\n{e}")

    def guardar_gradcam(self):
        if self.img_gradcam_pil is None:
            messagebox.showwarning("Sin Grad-CAM", "Primero genera Grad-CAM.")
            return

        sugerido = "gradcam_overlay.png"
        if self.ruta_actual:
            base = Path(self.ruta_actual).stem
            sugerido = f"{base}_gradcam.png"

        out = filedialog.asksaveasfilename(
            title="Guardar Grad-CAM",
            defaultextension=".png",
            initialfile=sugerido,
            filetypes=[("PNG", "*.png"), ("JPG", "*.jpg *.jpeg"), ("Todos", "*.*")]
        )
        if not out:
            return

        try:
            self.img_gradcam_pil.save(out)
            self._set_status(f"Guardado: {out}")
            messagebox.showinfo("Listo", f"Grad-CAM guardado en:\n{out}")
        except Exception as e:
            self._set_status(f"Error guardando: {e}")
            messagebox.showerror("Error", f"No se pudo guardar:\n\n{e}")


# ======================================================
# BLOQUE 6: MAIN
# ======================================================
# - Verifica que exista el archivo de pesos por defecto
# - Inicia la app
# ======================================================

def main():
    # Nota: si no existe la ruta por defecto, la UI aún abre y puedes seleccionar pesos manualmente
    if not Path(RUTA_PESOS).exists():
        print(f"[WARN] No se encontró el archivo de pesos en:\n{RUTA_PESOS}")

    root = tk.Tk()
    app = AplicacionClasificador(root)
    root.mainloop()


if __name__ == "__main__":
    main()
