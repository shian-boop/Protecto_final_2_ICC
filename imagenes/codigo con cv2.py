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
imagenes_dataset = misDatos.images    # imágenes 8x8
datos_dataset = misDatos.data         # datos aplanados (1797, 64)
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
    imagen = cv2.imread(nombre_de_imagen, cv2.IMREAD_GRAYSCALE)

    # 2. Redimensionar a 8x8 píxeles
    imagen_8x8 = cv2.resize(imagen, (8, 8))

    # 3. Invertir colores de la imagen
    imagen_invertida = 255 - imagen_8x8

    # 4. Normalizar a rango 0-16
    # Dividimos entre 255 (para tener 0-1) y multiplicamos por 16
    imagen_procesada = (imagen_invertida / 255.0) * 16

    return imagen_procesada

# ============================================================================
# Paso 4: Calcular las distancias euclidianas con cada
# ============================================================================

def calcular_distancia_euclidiana(imagen1, imagen2):
    """ Formula:
    distancia = √[(p1-q1)² + (p2-q2)² + ... + (p64-q64)²]
    """
    return np.linalg.norm(imagen1.flatten() - imagen2.flatten())
    #linalg es una funcion de numpy para operaciones con vectores

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
    k_vecinos = distancias[:3]

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

# ============================================================================
# PASO 7: PROGRAMA PRINCIPAL
# ============================================================================

print("\n" + "="*70)
print("SISTEMA DE RECONOCIMIENTO DE DÍGITOS ESCRITOS A MANO")
print("="*70)

# Solicitar datos al usuario
ruta = input("\nRuta de la imagen: ")
etiqueta_real = int(input("¿Qué número es realmente? (0-9): "))

# Procesar la imagen
print("\n[1] Procesando imagen...")
mi_imagen = procesar_imagen(ruta)
print("    ✓ Imagen procesada correctamente")

# Encontrar los 3 vecinos más cercanos
print("\n[2] Buscando los 3 vecinos más cercanos...")
vecinos = encontrar_k_vecinos(mi_imagen, datos_dataset, etiquetas_dataset, k=3)
print("    ✓ Vecinos encontrados")

# Extraer solo las etiquetas
etiquetas_vecinos = []
for tupla in vecinos:
    etiquetas_vecinos.append(tupla[2])

# Clasificar el dígito
print("\n[3] Clasificando dígito...")
prediccion = clasificar_el_digito(mi_imagen, vecinos)

# Mostrar resultados
print("\n" + "="*70)
print("RESULTADOS")
print("="*70)
print(f"\nNúmero real: {etiqueta_real}")
print(f"Targets de los 3 vecinos más cercanos: {etiquetas_vecinos}")

print("\n" + "-"*70)
print("Detalles de los vecinos:")
for i, (indice, distancia, etiqueta) in enumerate(vecinos, 1):
    print(f"  Vecino {i}: Etiqueta={etiqueta}, Distancia={distancia:.2f}")
print("-"*70)

print(f"\n🤖 Soy la inteligencia artificial, y he detectado que el dígito")
print(f"   ingresado corresponde al número {prediccion}")

if prediccion == etiqueta_real:
    print("\n✅ ¡CORRECTO! La IA clasificó correctamente el dígito")
else:
    print(f"\n❌ INCORRECTO. El número real era {etiqueta_real}")

print("\n" + "="*70)
