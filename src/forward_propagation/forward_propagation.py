# ============================================================
# Universidad Politécnica de Santa Rosa Jáuregui
#
# Materia: Redes Neuronales
# Profesor: Jesús Salvador López Ortega
# Grupo: IRC03
# Archivo: forward_propagation.py
# Descripción: Implementación del proceso de propagación hacia adelante
#              (forward propagation) para una red neuronal lineal.
# ============================================================

import sys, os, random
import numpy as np
from perceptron import InputData, Perceptron

#############################################################################################################################
# Algoritmo de propagación hacia adelante en una red neuronal                                                               #
#                                                                                                                           #
# Estructura general:                                                                                                       #
# - Entradas numéricas que alimentan la red.                                                                                #
# - Varias capas intermedias con un número determinado de perceptrones.                                                     #
# - Una capa final que produce la salida de la red.                                                                         #
#                                                                                                                           #
# Secuencia del algoritmo:                                                                                                  #
# 1. Convertir las entradas en objetos InputData.                                                                           #
# 2. En cada capa:                                                                                                          #
#    - Cada neurona recibe todas las salidas de la capa anterior como entradas.                                             #
#    - Ejecuta su función de activación y genera su salida.                                                                 #
# 3. La última neurona de salida recibe las activaciones de la última capa oculta.                                          #
# 4. El resultado final de la red es la activación de esa neurona de salida.                                                #
#############################################################################################################################

def forward_propagation_network(inputs: np.ndarray, perceptrons: int, layers: int) -> float:
    """
    Ejecuta una red neuronal de propagación lineal hacia adelante.
    """
    
    # Preparar los valores de entrada para la primera capa
    valores_actuales = list(map(float, inputs))

    # ==========================================================
    # Ciclo de propagación a través de las capas ocultas
    # ==========================================================
    for _ in range(layers):
        salidas_capa = []

        # Crear y ejecutar cada neurona de la capa
        for _ in range(perceptrons):
            datos_entrada = [InputData(x=v) for v in valores_actuales]
            neurona = Perceptron(inputs=datos_entrada, b=np.random.randn() * 0.01)
            neurona.run()
            salidas_capa.append(neurona.a)

        # Las salidas de esta capa se convierten en entradas para la siguiente
        valores_actuales = salidas_capa

    # ==========================================================
    # Capa de salida — combinación de las últimas activaciones
    # ==========================================================
    datos_finales = [InputData(x=v) for v in valores_actuales]
    neurona_salida = Perceptron(inputs=datos_finales, b=np.random.randn() * 0.01)
    neurona_salida.run()
 
    # Retornar el resultado final de la red
    return float(neurona_salida.a)
