# Especificación Técnica de la Arquitectura y Algoritmos

Este documento describe con rigor matemático y computacional los módulos que componen el sistema de clasificación de citología cervical.

---

## 1. Normalización de Color de Reinhard en CIELAB

Las variaciones en la preparación de frotis de Papanicolaou (tiempo de fijación, concentración de hematoxilina, eosina y naranja G) inducen desviaciones en la distribución de color. La normalización de Reinhard traslada las estadísticas de una imagen de referencia estándar a cada muestra.

1. **Transformación de Espacio:** Se convierte la imagen $I_{\text{BGR}}$ al espacio perceptual $I_{\text{LAB}}$.
2. **Estandarización por Canal:** Para cada canal $k \in \{L, A, B\}$:
   $$I_k^* = \frac{I_k - \mu_k}{\sigma_k + \epsilon} \cdot \sigma_k^{\text{target}} + \mu_k^{\text{target}}$$
   Donde:
   - $\mu^{\text{target}} = [145, 128, 128]$
   - $\sigma^{\text{target}} = [20, 12, 12]$
   - $\epsilon = 10^{-6}$ para estabilidad numérica.
3. **Reconversión:** Se recorta al rango $[0, 255]$ y se convierte de nuevo a RGB.

---

## 2. Segmentación de Tejido y Recorte Seguro (Tissue-Aware Cropping)

Para evitar recortes vacíos o enfocados en detritos celulares aislados (*núcleo suelto*):

1. **Detección de Crominancia y Luminosidad:**
   $$\text{Chroma} = \sqrt{(A - 128)^2 + (B - 128)^2}$$
   $$\text{Mask}_{\text{loose}} = (\text{Chroma} > 6) \lor (L < 245)$$
2. **Operaciones Morfológicas:**
   - Filtro de mediana $5 \times 5$.
   - Apertura morfológica con kernel $3 \times 3$ (elimina ruido impulsivo).
   - Clausura morfológica con kernel $7 \times 7$ (unifica citoplasma contiguo).
3. **Análisis de Componentes Conectados (8-conectividad):**
   - Se filtran componentes con área menor a $2,500$ píxeles.
   - Se selecciona el componente de mayor área contigua.
4. **Margen de Seguridad (Padding):**
   $$\text{pad} = 0.15 \times \max(W_{\text{comp}}, H_{\text{comp}})$$
   Garantiza la preservación del citoplasma que circunda el núcleo atípico.

---

## 3. Transfer Learning con EfficientNetB0

EfficientNetB0 utiliza escalado compuesto para optimizar profundidad ($d$), anchura ($w$) y resolución ($r$):
$$\text{profundidad: } d = \alpha^\phi, \quad \text{anchura: } w = \beta^\phi, \quad \text{resolución: } r = \gamma^\phi$$
sujeto a $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ y $\alpha \ge 1, \beta \ge 1, \gamma \ge 1$.

### Cabeza de Clasificación:
- **Global Average Pooling 2D:** Reduce tensores espaciales $(7, 7, 1280)$ a un vector unidimensional de $1280$.
- **Capa Densa (256 neuronas, ReLU):** Representación de características de alto nivel.
- **Dropout (0.40):** Regularización frente a sobreajuste.
- **Capa Softmax (3 clases):**
  $$P(Y = c | X) = \frac{e^{z_c}}{\sum_{j=1}^3 e^{z_j}}, \quad c \in \{\text{HSIL}, \text{LSIL}, \text{NILM}\}$$

---

## 4. Inteligencia Artificial Explicable (Grad-CAM)

Grad-CAM genera un mapa de activación visual para la clase de interés $c$:

1. Se obtiene el gradiente de la puntuación $y^c$ respecto a los mapas de activación $A^k$ de la última capa convolucional (`top_conv`):
   $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$
2. Se ponderan los mapas de características mediante combinación lineal y se aplica una rectificación ReLU para retener solo características positivas:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
3. El mapa se reescala a $224 \times 224$, se aplica el mapa de color JET y se superpone sobre la imagen preprocesada con un factor $\alpha = 0.55$.
