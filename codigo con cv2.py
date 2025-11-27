# ============================================================================
# PASO 1: Importar librerías
# ============================================================================
import numpy as np
from sklearn import datasets
import cv2

# ============================================================================
# PASO 2: Cargar el dataset
# ============================================================================

misDatos = datasets.load_digits()
imagenes_dataset = misDatos["images"]
datos_dataset = misDatos["data"]
etiquetas_dataset = misDatos["target"]

# ============================================================================
# PASO 3: Procesar imagenes
# ============================================================================
def procesar_imagen(nombre_de_imagen):
    """ Pasos:
    1. Leer imagen en escala de grises
    2. Redimensionar a 8x8 píxeles
    3. Invertir colores de la imagen
    4. Convertir a escala de 1 a 16
    """
    img_array = cv2.imread(nombre_de_imagen, cv2.IMREAD_GRAYSCALE)
    nueva_img = cv2.resize(img_array, (8, 8))
    i = 0
    while i < 8:
        j = 0
        while j < 8:
            nueva_img[i][j] = 255 - nueva_img[i][j]
            j = j + 1
        i = i + 1

    i = 0
    while i < 8:
        j = 0
        while j < 8:
            nueva_img[i][j] = nueva_img[i][j] / 255 * 16
            j = j + 1
        i = i + 1

    print(nueva_img)
    return nueva_img

# ============================================================================
# PASO 4: Calcular distancia euclidiana
# ============================================================================
def calcular_distancia(vector1, vector2):
    diferencias_al_cuadrado = (vector1 - vector2)**2
    suma = sum(diferencias_al_cuadrado)
    distancia = suma**0.5
    return distancia

# ==================================================
# PASO 6: ENCONTRAR LOS K VECINOS MÁS CERCANOS
# ====================================================
def encontrar_k_vecinos(imagen, datos_dataset, etiquetas_dataset, k=3):
    mi_vector = imagen.flatten()
    distancias = []
    for i in range(len(datos_dataset)):
        vector_dataset = datos_dataset[i]
        distancia = calcular_distancia(mi_vector, vector_dataset)
        distancias.append((distancia, etiquetas_dataset[i]))

    distancias.sort()
    k_vecinos = distancias[:k]

    return k_vecinos

# ============================================================================
# paso 7: clasificar los digitos q ingresamos
# ============================================================================
def clasificar_el_digito(mi_imagen, vecinos_3):
        etiquetas = []
        for vecino in vecinos_3:
            etiquetas.append(vecino[1])

        mejor_numero = 0
        mejor_cantidad = 0

        for numero in range(10):
            cantidad = etiquetas.count(numero)
            if cantidad > mejor_cantidad:
                mejor_cantidad = cantidad
                mejor_numero = numero

        if mejor_cantidad >= 2:
            return mejor_numero
        else:
            vecinos_5 = encontrar_k_vecinos(mi_imagen, datos_dataset, etiquetas_dataset, k=5)
            etiquetas_5 = []
            for vecino in vecinos_5:
                etiquetas_5.append(vecino[1])

            mejor_numero_5 = 0
            mejor_cantidad_5 = 0
            for numero in range(10):
                cantidad = etiquetas_5.count(numero)
                if cantidad > mejor_cantidad_5:
                    mejor_cantidad_5 = cantidad
                    mejor_numero_5 = numero

            return mejor_numero_5


#ejecutar funciones con las
lista=["cero","uno","dos","tres","cuatro","cinco","seis","siete","ocho","nueve"]
for i in lista:
    ruta= "./imagenes/"+i+".png"
    imagen = procesar_imagen(ruta)
    vecinos_3 = encontrar_k_vecinos(imagen, datos_dataset, etiquetas_dataset, k=3)
    resultado = clasificar_el_digito(imagen, vecinos_3)

    print("El dígito detectado es:", resultado)

