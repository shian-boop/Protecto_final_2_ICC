# ============================================================================
# PASO 1: Importar librerías
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import cv2
from collections import Counter

# ============================================================================
# PASO 2: Cargar el dataset
# ============================================================================

misDatos = datasets.load_digits()
imagenes_dataset = misDatos.images
datos_dataset = misDatos.data
etiquetas_dataset = misDatos.target

# ============================================================================
# PASO 3: Procesar imagenes
# ============================================================================

def procesar_imagen(nombre_de_imagen):
    """
    Pasos:
    1. Leer imagen en blanco y negro
    2. Reducirla a 8x8 pixeles
    3. Invertir la escala (los colores)
    4. Reducir los valores a un rango de 0-16
    """
    # 1. Leer imagen
    img_array = cv2.imread(nombre_de_imagen, cv2.IMREAD_GRAYSCALE)

    # 2. Redimensionar a 8x8 píxeles
    nueva_img = cv2.resize(img_array, (8, 8))

    nueva_img = (255 - nueva_img) / 255 * 16
    # 3. Invertir colores de la imagen
    imagen_invertida = 255 - nueva_img

    # 4. Normalizar a rango 0-16
    # Dividimos entre 255 (para tener 0-1) y multiplicamos por 16
    imagen_procesada = (imagen_invertida / 255.0) * 16

    return imagen_procesada

# ============================================================================
# Paso 4: Calcular las distancias euclidianas con cada
# ============================================================================


# ==================================================
# PASO 6: ENCONTRAR LOS K VECINOS MÁS CERCANOS
# ====================================================

def encontrar_k_vecinos(mi_imagen, datos_dataset, etiquetas_dataset, k=3):

    # aplanar la imagen (convertir 8x8 a vector de 64)
    mi_vector = mi_imagen.flatten()

    distancias = [] #aqui guardaremos todas las ditancias para luego comparar

    # calcular distancia con cada imagen del dataset digits
    for i in range(len(datos_dataset)):
        vector_dataset = datos_dataset[i]

        # calcular distancia euclidiana con numpy
        distancia = np.linalg.norm(mi_vector - vector_dataset)

        # agregamos a nuestra lista "distancias" el índice en el dataset, la distancia y la etiqueta del digito
        distancias.append((i, distancia, etiquetas_dataset[i]))

    # Ordenar por distancia (de menor a mayor)
    distancias.sort(key=lambda x: x[1]) #usamos lambda para ordenar por
    # el segundo elemento de la lista que son las distancias

    # Tomar los K primeros (los más cercanos), en este caso tomaremos los 3 primeros knn
    k_vecinos = distancias[:k]

    return k_vecinos

# ============================================================================
# paso 7: clasificar los digitos q ingresamos
# ============================================================================

def clasificar_el_digito(mi_imagen, vecinos_3):
    #clasificamos en base a los 3 vecinos mas cercanos
    # en caso de q los 3 targets sean diferentes nuestra solucion sera usar 5 vecinos
    # mas cercanos y elegir en base a ello

    # extraer solo las etiquetas de los 3 vecinos
    etiquetas_3 = []
    for tupla in vecinos_3:
        etiquetas_3.append(tupla[2])  # tupla[2] es la etiqueta

    # contar cuántas veces aparece cada etiqueta
    conteo = Counter(etiquetas_3)
    mas_comun, frecuencia = conteo.most_common(1)[0]

    if frecuencia >= 2:
        #si haay mayoría (2 o 3 iguales) usamos esa etiqueta
        return mas_comun
    else:
        # si los 3 son diferences buscaremos 5 vecinos para más precision
        print("    (Los 3 vecinos son diferentes, expandiendo a 5 vecinos...)")
        vecinos_5 = encontrar_k_vecinos(mi_imagen, datos_dataset, etiquetas_dataset, k=5)

        # extraer etiquetas de los 5 vecinos
        etiquetas_5 = []
        for tupla in vecinos_5:
            etiquetas_5.append(tupla[2])

        # hacer votación con 5 vecinos
        conteo_5 = Counter(etiquetas_5)
        ganador, _ = conteo_5.most_common(1)[0]
        return ganador

lista=["cero","uno","dos","tres","cuatro","cinco","seis","siete","ocho","nueve"]
for i in lista:
    ruta= "./imagenes/"+i+".png"

    imagen = procesar_imagen(ruta)

    # 2. Calculas los 3 vecinos
    vecinos_3 = encontrar_k_vecinos(imagen, datos_dataset, etiquetas_dataset, k=3)

    # 3. Clasificas usando esos vecinos
    resultado = clasificar_el_digito(imagen, vecinos_3)

    print("El dígito detectado es:", resultado)

#