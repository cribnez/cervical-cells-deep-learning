"""
Clinical Decision Support System - Graphical User Interface (Tkinter) with Grad-CAM XAI.
"""
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from pathlib import Path
from PIL import Image, ImageTk
import numpy as np

from .config import (
    CLASES,
    COLORES,
    DEFAULT_NILM_THRESHOLD,
    GRADCAM_LAYER_NAME,
    DEFAULT_WEIGHTS_PATH,
)
from .preprocessing import preprocess_single_image
from .model import (
    crear_modelo_functional,
    cargar_pesos_pickle,
    generar_gradcam_superpuesto_x,
)


class AplicacionClasificador:
    def __init__(self, root, default_weights: Path = DEFAULT_WEIGHTS_PATH):
        self.root = root
        self.root.title("CAD Citología Cervical — Sistema Bethesda (HSIL / LSIL / NILM) + Grad-CAM")
        self.root.geometry("1060x730")
        self.root.minsize(980, 680)

        self.default_weights = Path(default_weights)
        self.model = None
        self.base_model = None

        self.ruta_actual = None
        self.idx_clase_actual = None

        self.img_original_pil = None
        self.img_preproc_pil = None
        self.img_gradcam_pil = None
        self.x_actual = None
        self.prob_actual = None

        self.var_threshold = tk.DoubleVar(value=float(DEFAULT_NILM_THRESHOLD))

        self._setup_theme()
        self._build_layout()
        self._cargar_modelo_inicial(self.default_weights)

    def _setup_theme(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        self.root.configure(bg="#0B1220")

        style.configure("App.TFrame", background="#0B1220")
        style.configure("Card.TFrame", background="#111A2E")

        style.configure("HeaderTitle.TLabel", background="#0B1220", foreground="white",
                        font=("Arial", 16, "bold"))
        style.configure("HeaderSubtitle.TLabel", background="#0B1220", foreground="#9CA3AF",
                        font=("Arial", 10))
        style.configure("PanelTitle.TLabel", background="#111A2E", foreground="white",
                        font=("Arial", 12, "bold"))
        style.configure("Info.TLabel", background="#111A2E", foreground="#CBD5E1",
                        font=("Arial", 10))
        style.configure("Status.TLabel", background="#0B1220", foreground="#9CA3AF",
                        font=("Arial", 9))

        style.configure("Primary.TButton", font=("Arial", 10, "bold"))
        style.configure("Secondary.TButton", font=("Arial", 10))

        style.configure("TNotebook", background="#111A2E", borderwidth=0)
        style.configure("TNotebook.Tab", background="#1F2A44", foreground="white",
                        padding=(12, 6), font=("Arial", 10, "bold"))
        style.map("TNotebook.Tab",
                  background=[("selected", "#3B82F6")],
                  foreground=[("selected", "white")])

        # Progressbar styles per class
        style.configure("HSIL.Horizontal.TProgressbar", troughcolor="#1F2A44", background="#EF4444")
        style.configure("LSIL.Horizontal.TProgressbar", troughcolor="#1F2A44", background="#F59E0B")
        style.configure("NILM.Horizontal.TProgressbar", troughcolor="#1F2A44", background="#10B981")

    def _build_layout(self):
        header = ttk.Frame(self.root, style="App.TFrame")
        header.pack(fill="x", padx=18, pady=(14, 8))

        ttk.Label(
            header,
            text="Sistema CAD: Diagnóstico Automatizado de Lesiones Cervicales (EfficientNetB0 + XAI)",
            style="HeaderTitle.TLabel"
        ).pack(anchor="w")

        ttk.Label(
            header,
            text="Triaje Clínico Multiclase (Bethesda) • Normalización Reinhard • Grad-CAM Explicable",
            style="HeaderSubtitle.TLabel"
        ).pack(anchor="w")

        body = ttk.Frame(self.root, style="App.TFrame")
        body.pack(expand=True, fill="both", padx=18, pady=8)
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=2)
        body.grid_rowconfigure(0, weight=1)

        # Left Panel: Image Views
        self.left_card = ttk.Frame(body, style="Card.TFrame")
        self.left_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.left_card.grid_columnconfigure(0, weight=1)
        self.left_card.grid_rowconfigure(1, weight=1)

        ttk.Label(self.left_card, text="Inspección Visual Citológica", style="PanelTitle.TLabel").grid(
            row=0, column=0, sticky="w", padx=14, pady=(12, 6)
        )

        self.notebook = ttk.Notebook(self.left_card)
        self.notebook.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))

        self.tab_original = ttk.Frame(self.notebook)
        self.tab_preproc = ttk.Frame(self.notebook)
        self.tab_gradcam = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_original, text="Original")
        self.notebook.add(self.tab_preproc, text="Preprocesada (Reinhard + Recorte)")
        self.notebook.add(self.tab_gradcam, text="Grad-CAM (Mapa de Calor)")

        self.lbl_img_original = tk.Label(self.tab_original, bg="#0B1220", fg="#9CA3AF", text="Sin imagen cargada")
        self.lbl_img_preproc = tk.Label(self.tab_preproc, bg="#0B1220", fg="#9CA3AF", text="Sin imagen procesada")
        self.lbl_img_gradcam = tk.Label(self.tab_gradcam, bg="#0B1220", fg="#9CA3AF", text="Sin mapa Grad-CAM")

        for lbl in (self.lbl_img_original, self.lbl_img_preproc, self.lbl_img_gradcam):
            lbl.pack(expand=True, fill="both", padx=8, pady=8)

        # Right Panel: Results & Clinical Controls
        self.right_card = ttk.Frame(body, style="Card.TFrame")
        self.right_card.grid(row=0, column=1, sticky="nsew")
        self.right_card.grid_columnconfigure(0, weight=1)

        ttk.Label(self.right_card, text="Resultados & Triaje Clínico", style="PanelTitle.TLabel").grid(
            row=0, column=0, sticky="w", padx=14, pady=(12, 6)
        )

        self.lbl_model_status = ttk.Label(self.right_card, text="Modelo: (cargando...)", style="Info.TLabel")
        self.lbl_model_status.grid(row=1, column=0, sticky="w", padx=14)

        self.lbl_pesos = ttk.Label(self.right_card, text=f"Pesos: {self.default_weights.name}", style="Info.TLabel", wraplength=360)
        self.lbl_pesos.grid(row=2, column=0, sticky="w", padx=14, pady=(4, 10))

        btns = tk.Frame(self.right_card, bg="#111A2E")
        btns.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 10))
        btns.grid_columnconfigure(0, weight=1)
        btns.grid_columnconfigure(1, weight=1)

        self.btn_select_img = ttk.Button(btns, text="📁 Cargar Imagen", style="Primary.TButton",
                                         command=self.seleccionar_imagen)
        self.btn_select_img.grid(row=0, column=0, sticky="ew", padx=(0, 6), ipady=5)

        self.btn_select_pesos = ttk.Button(btns, text="🧠 Cargar Pesos (.pkl)", style="Secondary.TButton",
                                           command=self.seleccionar_pesos)
        self.btn_select_pesos.grid(row=0, column=1, sticky="ew", padx=(6, 0), ipady=5)

        # Probability Bars
        self.labels_prob = {}
        self.barras_prob = {}
        prob_frame = tk.Frame(self.right_card, bg="#111A2E")
        prob_frame.grid(row=4, column=0, sticky="ew", padx=14, pady=(8, 8))
        prob_frame.grid_columnconfigure(1, weight=1)

        for r, clase in enumerate(CLASES):
            lbl = ttk.Label(prob_frame, text=f"{clase}", style="Info.TLabel")
            lbl.grid(row=r, column=0, sticky="w", pady=5)

            bar = ttk.Progressbar(prob_frame, mode="determinate", length=200,
                                  style=f"{clase}.Horizontal.TProgressbar")
            bar.grid(row=r, column=1, sticky="ew", padx=(10, 10), pady=5)

            val = ttk.Label(prob_frame, text="0.00%", style="Info.TLabel")
            val.grid(row=r, column=2, sticky="e", pady=5)

            self.barras_prob[clase] = bar
            self.labels_prob[clase] = val

        self.lbl_pred = tk.Label(self.right_card, text="Diagnóstico: —", bg="#111A2E",
                                 fg="white", font=("Arial", 15, "bold"))
        self.lbl_pred.grid(row=5, column=0, sticky="ew", padx=14, pady=(8, 4))

        self.lbl_triaje = tk.Label(self.right_card, text="Conducta de Triaje: —", bg="#111A2E",
                                   fg="#CBD5E1", font=("Arial", 11, "bold"))
        self.lbl_triaje.grid(row=6, column=0, sticky="ew", padx=14, pady=(0, 8))

        # Triage Slider
        thr_frame = tk.Frame(self.right_card, bg="#111A2E")
        thr_frame.grid(row=7, column=0, sticky="ew", padx=14, pady=(4, 6))
        thr_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(thr_frame, text="Umbral Clínico de Negatividad (NILM):", style="Info.TLabel").grid(row=0, column=0, sticky="w")
        self.lbl_thr_val = ttk.Label(thr_frame, text=f"{self.var_threshold.get():.2f}", style="Info.TLabel")
        self.lbl_thr_val.grid(row=0, column=2, sticky="e")

        self.scale_thr = ttk.Scale(thr_frame, from_=0.50, to=0.99, variable=self.var_threshold,
                                   command=self._on_threshold_change)
        self.scale_thr.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(4, 0))

        # Action Buttons
        action_frame = tk.Frame(self.right_card, bg="#111A2E")
        action_frame.grid(row=8, column=0, sticky="ew", padx=14, pady=(10, 8))
        action_frame.grid_columnconfigure(0, weight=1)
        action_frame.grid_columnconfigure(1, weight=1)

        self.btn_make_gradcam = ttk.Button(action_frame, text="🔥 Generar Grad-CAM",
                                           style="Primary.TButton", command=self.generar_gradcam,
                                           state="disabled")
        self.btn_make_gradcam.grid(row=0, column=0, sticky="ew", padx=(0, 6), ipady=4)

        self.btn_save_gradcam = ttk.Button(action_frame, text="💾 Exportar Mapa",
                                           style="Secondary.TButton", command=self.guardar_gradcam,
                                           state="disabled")
        self.btn_save_gradcam.grid(row=0, column=1, sticky="ew", padx=(6, 0), ipady=4)

        # Informative Note
        self.lbl_info = tk.Label(
            self.right_card,
            text=(
                "• Estandarización: Normalización de color Reinhard en espacio CIELAB.\n"
                "• Segmentación: Detección tisular y recorte seguro centrado en celularidad.\n"
                "• Triaje Seguro: Se clasifica como NEGATIVO solo si p(NILM) supera el umbral configurado."
            ),
            bg="#111A2E",
            fg="#9CA3AF",
            font=("Arial", 8),
            justify="left",
            wraplength=380
        )
        self.lbl_info.grid(row=9, column=0, sticky="w", padx=14, pady=(6, 10))

        self.status = ttk.Label(self.root, text="Listo.", style="Status.TLabel")
        self.status.pack(fill="x", padx=18, pady=(0, 8))

    def _set_status(self, msg: str):
        self.status.configure(text=msg)
        self.root.update_idletasks()

    def _set_image_on_label(self, lbl: tk.Label, img_pil: Image.Image):
        if img_pil is None:
            lbl.configure(image="", text="Sin imagen")
            lbl.image = None
            return

        w = max(480, self.left_card.winfo_width() - 60)
        h = max(480, self.left_card.winfo_height() - 140)

        img_show = img_pil.copy()
        img_show.thumbnail((w, h))
        foto = ImageTk.PhotoImage(img_show)
        lbl.configure(image=foto, text="")
        lbl.image = foto

    def _update_view_tabs(self):
        self._set_image_on_label(self.lbl_img_original, self.img_original_pil)
        self._set_image_on_label(self.lbl_img_preproc, self.img_preproc_pil)
        self._set_image_on_label(self.lbl_img_gradcam, self.img_gradcam_pil)

    def _cargar_modelo_inicial(self, ruta_pesos: Path):
        ruta_pesos = Path(ruta_pesos)
        if not ruta_pesos.exists():
            self.model, self.base_model = crear_modelo_functional()
            self.lbl_model_status.configure(text="Modelo: ⚠️ Inicializado con pesos ImageNet (Cargar .pkl para fine-tuning)")
            self.lbl_pesos.configure(text=f"Pesos: No cargados ({ruta_pesos.name} no encontrado)")
            self._set_status("Modelo listo con pesos ImageNet.")
            return

        try:
            self._set_status(f"Cargando pesos desde {ruta_pesos.name}...")
            self.model, self.base_model = crear_modelo_functional()
            self.model = cargar_pesos_pickle(self.model, self.base_model, ruta_pesos)
            self.lbl_model_status.configure(text="Modelo: ✓ Pesos entrenados cargados correctamente")
            self.lbl_pesos.configure(text=f"Pesos: {ruta_pesos.name}")
            self._set_status("Modelo cargado con éxito.")
        except Exception as e:
            self.lbl_model_status.configure(text="Modelo: ✗ Error al cargar pesos")
            self._set_status(f"Error: {e}")

    def seleccionar_pesos(self):
        ruta = filedialog.askopenfilename(
            title="Seleccionar archivo de pesos (.pkl)",
            filetypes=[("Pickle", "*.pkl"), ("Todos", "*.*")]
        )
        if not ruta:
            return
        self._cargar_modelo_inicial(Path(ruta))

    def _on_threshold_change(self, _evt=None):
        self.lbl_thr_val.configure(text=f"{self.var_threshold.get():.2f}")
        if self.prob_actual is not None:
            self._actualizar_triaje(self.prob_actual)

    def seleccionar_imagen(self):
        ruta = filedialog.askopenfilename(
            title="Seleccionar imagen citológica (microfotografía)",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("Todos", "*.*")]
        )
        if ruta:
            self.analizar_imagen(ruta)

    def analizar_imagen(self, ruta):
        if self.model is None:
            messagebox.showwarning("Modelo no disponible", "Inicialice o cargue los pesos del modelo.")
            return

        self.ruta_actual = ruta
        self.img_gradcam_pil = None
        self.btn_save_gradcam.config(state="disabled")

        try:
            self._set_status("Aplicando Reinhard y recorte seguro...")
            x_tensor, img_preproc_pil, img_original_pil = preprocess_single_image(ruta)
            self.x_actual = x_tensor
            self.img_preproc_pil = img_preproc_pil
            self.img_original_pil = img_original_pil
            self._update_view_tabs()
            self.notebook.select(self.tab_original)
        except Exception as e:
            messagebox.showerror("Error", f"Fallo al preprocesar imagen:\n{e}")
            return

        try:
            self._set_status("Ejecutando inferencia...")
            prob = self.model.predict(self.x_actual, verbose=0)[0]
            self.prob_actual = prob

            for i, clase in enumerate(CLASES):
                p = float(prob[i]) * 100.0
                self.barras_prob[clase]["value"] = p
                self.labels_prob[clase].configure(text=f"{p:.2f}%")

            idx_max = int(np.argmax(prob))
            clase_predicha = CLASES[idx_max]
            confianza = float(prob[idx_max]) * 100.0
            self.idx_clase_actual = idx_max

            self.lbl_pred.configure(
                text=f"Diagnóstico: {clase_predicha} ({confianza:.1f}%)",
                fg=COLORES[clase_predicha]
            )

            self._actualizar_triaje(prob)
            self.btn_make_gradcam.config(state="normal")
            self._set_status("Predicción lista. Puede generar Grad-CAM.")
        except Exception as e:
            messagebox.showerror("Error", f"Error en inferencia:\n{e}")

    def _actualizar_triaje(self, prob_vec):
        thr = float(self.var_threshold.get())
        p_nilm = float(prob_vec[2])

        if p_nilm >= thr:
            triage = f"✅ TRIAJE: PROBABLE NEGATIVO (NILM) [p={p_nilm:.2f} ≥ {thr:.2f}]"
            color = "#10B981"
        else:
            triage = f"⚠️ TRIAJE: REMISIÓN A COLPOSCOPIA / REVISIÓN [p={p_nilm:.2f} < {thr:.2f}]"
            color = "#F59E0B"

        self.lbl_triaje.configure(text=triage, fg=color)

    def generar_gradcam(self):
        if self.model is None or self.x_actual is None or self.idx_clase_actual is None:
            return

        try:
            self._set_status("Calculando gradientes para Grad-CAM...")
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
            self._set_status("Mapa de atención Grad-CAM visualizado.")
        except Exception as e:
            messagebox.showerror("Error Grad-CAM", f"No se pudo generar Grad-CAM:\n{e}")

    def guardar_gradcam(self):
        if self.img_gradcam_pil is None:
            return

        sugerido = "gradcam_overlay.png"
        if self.ruta_actual:
            sugerido = f"{Path(self.ruta_actual).stem}_gradcam.png"

        out = filedialog.asksaveasfilename(
            title="Guardar Mapa Grad-CAM",
            defaultextension=".png",
            initialfile=sugerido,
            filetypes=[("PNG", "*.png"), ("JPG", "*.jpg"), ("Todos", "*.*")]
        )
        if out:
            self.img_gradcam_pil.save(out)
            messagebox.showinfo("Exportación Exitosa", f"Mapa Grad-CAM guardado en:\n{out}")


def main():
    root = tk.Tk()
    app = AplicacionClasificador(root)
    root.mainloop()


if __name__ == "__main__":
    main()
