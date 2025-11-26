# ============================================================================
# PASO 1: Importar librerías
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from PIL import Image

# ============================================================================
# PASO 2: Cargar el dataset
# ============================================================================

misDatos = datasets.load_digits()
imagenes_dataset = misDatos.images    # imágenes 8x8
datos_dataset = misDatos.data         # datos aplanados (1797, 64)
etiquetas_dataset = misDatos.target

# ============================================================================
# PASO 3: Procesar imagenes
# ============================================================================

def procesar_imagen_propia(ruta_imagen):
    """
    Pasos:
    1. Leer imagen en blanco y negro
    2. Reducirla a 8x8 pixeles
    3. Invertir la escala (los colores)
    4. Reducir los valores a un rango de 0-16
    """
    # 1. Leer imagen
    imagen = Image.open(ruta_imagen).convert('L')  # "L" convierte a escala de grises

    # 2. Redimensionar a 8x8
    imagen_8x8 = imagen.resize((8, 8))

    # 3. Convertir a array de numpy
    array_imagen = np.array(imagen_8x8)

    # 4. Invertir la escala de colores
    array_invertido = 255 - array_imagen

    # 5. Normalizar a rango 0-16 (como el dataset)
    array_normalizado = (array_invertido / 255.0) * 16

    return array_normalizado

# ============================================================================
# PASO 4: Calcular distancia euclidiana
# ============================================================================

def calcular_distancia_euclidiana(imagen1, imagen2):
    # usando funciones de numpy
    return np.linalg.norm(imagen1.flatten() - imagen2.flatten())
