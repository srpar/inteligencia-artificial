import cv2 # Se importa la libreria OpenCV para el procesamiento de imagenes
import numpy as np # Se importa la libreria numpy con el alias np. Aporta soporte para matrices y diversas funciones matemáticas 
import matplotlib.pyplot as plt # Se importa un modulo de la libreria pyplot con un alias que permite generar graficos
from scipy.spatial import distance # Se importa un modulo de la libreria scipy para facilitar calculo de distancias en espacios multidimensionales

# Se carga la imagen del caso del aro en la que se debe identificar la ubicacion en coordenadas del centro del aro
# Metodo de openCV para leer la imagen
imagen = cv2.imread('imagen-aro.jpg')

# Se convierte la imagen cargada a escala de grises para facilitar el procesamiento
imagen_escala_grises = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Se aplica una umbralización a la imagen y se transforma en un binario
_, imagen_umbralizada = cv2.threshold(imagen_escala_grises, 128, 255, cv2.THRESH_BINARY)

# Se busca detectar los contornos de la imagen procesada
# RETR_EXTERNAL detecta los contornos externos
# CHAIN_APPROX_SIMPLE refeire a un metodo simple para detectar los contornos
contornos, _ = cv2.findContours(imagen_umbralizada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Se filtran los contornos que superen cierto umbral
contornos_filtrados = [contorno for contorno in contornos if cv2.contourArea(contorno) > 100]

# Se define la variable de numero de neuronas y se asigna el valor de 5
numero_neuronas = 5

# Se inicializa la matriz de pesos correspondiente a la red
matriz_pesos = np.zeros((numero_neuronas, numero_neuronas))

# Se convierte la imagen binaria en un modelo de Hopfield, con 2 valores posibles
imagen_binaria = (imagen_umbralizada / 255).reshape(-1) * 2 - 1

# Se define una función para reconocer patrones Hopfield
# Se utiliza la matriz de pesos ajustada para reconocer patrones de Hopfield.
# Se inicia un ciclo for para buscar reconocer el patron
# matriz_pesos de la red Hopfield se carga en np.array
# patron que se busca reconocer se genera en np.array
# iteraciones es la cantidad de pasos para la reconstruccion
def reconocer_patron(matriz_pesos, patron, iteraciones=5):
    for _ in range(iteraciones):
        nuevo_patron = np.sign(np.dot(patron, matriz_pesos))
        if np.array_equal(nuevo_patron, patron):
            return nuevo_patron
        patron = nuevo_patron
    return patron

# Se retoma y actualiza  la estructura del modelo de Hopfield para el caso
matriz_pesos = np.zeros((numero_neuronas, numero_neuronas))

# Ejemplo de como se podria comenzar a entrenar la red Hopfield con un patrón
# Idealmente se deberían cargar varias imagenes para que el sistema identifique ubicaciones del patrón del centro del aro
# En algunos casos el centro estara a una distancia de X e Y y en otros casos a otra pero pueden presentarse patrones y tendencias
# El siguiente patron muestra un posible ejemplo abreviado y simplificado
patron_arco = np.array([1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 
                        1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1,
                        1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1,
                        1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1,
                        1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1,
                        1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1,
                        1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1,
                        1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1])

# Se crea la matriz de pesos con las neuronas y patrones detectados
def crear_matriz_pesos(neuronas, patrones):
    matriz_pesos = np.zeros((neuronas, neuronas))

    for i in range(neuronas):
        for j in range(neuronas):
            if i != j:
                for patron in patrones:
                    matriz_pesos[i, j] += patron[i] * patron[j]

    return matriz_pesos

# Se retoma la función para reconocer si existe patron con la red Hopfield
def reconocer_patron(matriz_pesos, patron, iteraciones=5):
    for _ in range(iteraciones):
        nuevo_patron = np.sign(np.dot(patron, matriz_pesos))
        if np.array_equal(nuevo_patron, patron):
            return nuevo_patron
        patron = nuevo_patron
    return patron


# Se crea la matriz de pesos Hopfield actualizada
numero_neuronas = len(patron_arco) # Se actualiza a partir del patron_arco simulado de ejemplo
matriz_pesos = crear_matriz_pesos(numero_neuronas, [patron_arco])

# Se busca reconocer el patrón
patron_reconocido = reconocer_patron(matriz_pesos, patron_arco)

# Se verifica y se muestra en la terminal si el patron se pudo identificar o no
if np.array_equal(patron_reconocido, patron_arco):
    print("El patrón se ha reconocido correctamente.")
else:
    print("No se ha reconocido el patrón correctamente.")

# Se procesa la imagen para mostrar el Centro del aro
# Se busca identificar el área con el aro C. Para ello se arranca con los contornos filtrados la posible ubicacion del centro del aro C
# Solo se avanza si hay contornos filtrados asignados a la variable correspondiente
if contornos_filtrados: 
    # Se busca el contorno más grande disponible
    contorno_mas_grande = max(contornos_filtrados, key=cv2.contourArea)
    # Se hace el calculo de los momentos del contorno más grande que podria ser el centro del aro buscado
    momentos = cv2.moments(contorno_mas_grande)

    # Se avanza solo si el area m00 es mayor que 0.
    if momentos["m00"] != 0:
        # Se realiza el calculo de las coordenada del centro del area en X e Y
        centro_x = int(momentos["m10"] / momentos["m00"])
        centro_y = int(momentos["m01"] / momentos["m00"])
        # Se dibuja un circulo para identificar el centro del aro buscado
        # 7 es el Radio del circulo, (0, 0, 255) es el color naranja, -1 significa que sera un circulo solido
        cv2.circle(imagen, (centro_x, centro_y), 7, (255, 165, 0), -1)

# Se dibujan los contornos en la imagen 
cv2.drawContours(imagen, contornos_filtrados, -1, (255, 165, 0), 2)

# Se declaran variables para dibujar líneas desde el centro del aro para resaltar la ubicacion buscada
x_referencia = 240 
y_referencia = 200 

# Se dibuja la línea punteada en el eje X
cv2.line(imagen, (centro_x, centro_y), (x_referencia, centro_y), (255, 255, 255), 1, cv2.LINE_AA)

# Se dibuja la línea punteada en el eje Y
cv2.line(imagen, (centro_x, centro_y), (centro_x, y_referencia), (255, 255, 255), 1, cv2.LINE_AA)

# Se muestra la imagen con los contornos y el centro identificado
cv2.imshow('Imagen con Contornos', imagen)
cv2.waitKey(0) # Se espera a que el usuario presione una tecla o cierre la ventana emergente para continuar con la ejecucion del codigo

# Se muestra la imagen umbralizada 
cv2.imshow('Imagen Preprocesada', imagen_umbralizada)
cv2.waitKey(0)

# Se muestra la imagen original
plt.imshow(imagen_escala_grises, cmap='gray')
plt.show()

# Se cierran las ventanas emergentes abiertas con las imagenes
cv2.destroyAllWindows()