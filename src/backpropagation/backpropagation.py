# ============================================================
# Universidad Politécnica de Santa Rosa Jáuregui
#
# Materia: Redes Neuronales
# Profesor: Jesús Salvador López Ortega
# Grupo: IRC03
# Archivo: backpropagation.py
# Descripción: Implementación del algoritmo de aprendizaje mediante retropropagación.
# ============================================================

import sys, os, random
import numpy as np
from perceptron.input_data import InputData
from perceptron.perceptron import Perceptron

#############################################################################################################################
# Algoritmo de retropropagación (backpropagation) aplicado a una red neuronal lineal.                                       #
#                                                                                                                           #
# Propósito: Ajustar los pesos de la red en función del error entre la salida obtenida y la esperada.                       #
#                                                                                                                           #
# Flujo general:                                                                                                            #
# 1. Se calcula el error de la salida.                                                                                      #
# 2. Se obtiene el gradiente (delta) de la neurona de salida.                                                               #
# 3. Se propaga el error hacia las capas previas (ocultas).                                                                 #
# 4. Se actualizan los pesos y sesgos usando una tasa de aprendizaje.                                                       #
# 5. Se repite el proceso para todas las muestras de entrenamiento.                                                         #
#############################################################################################################################

def backpropagation_network(inputs: np.ndarray, perceptrons: int, layers: int) -> float:
    """
    Ejecuta una red neuronal de propagación lineal con ajuste de pesos mediante backpropagation.
    """

    # --- Inicialización de datos base ---
    entradas = [float(valor) for valor in inputs]
    lr = 0.1  # tasa de aprendizaje

    # Determinar salida esperada (simulación de lógica OR)
    salida_deseada = 1.0 if any(v > 0 for v in entradas) else 0.0

    # --- 1. Construcción de la red (propagación hacia adelante) ---
    capa_actual = entradas
    red_completa = []

    for _ in range(layers):
        capa_salida = []
        capa_neuronas = []
        for _ in range(perceptrons):
            datos_neurona = [InputData(x=valor) for valor in capa_actual]
            neurona = Perceptron(inputs=datos_neurona, b=np.random.randn() * 0.01)
            neurona.run()
            capa_salida.append(neurona.a)
            capa_neuronas.append(neurona)
        red_completa.append(capa_neuronas)
        capa_actual = capa_salida

    # Capa de salida
    datos_salida = [InputData(x=valor) for valor in capa_actual]
    salida_neurona = Perceptron(inputs=datos_salida, b=np.random.randn() * 0.01)
    salida_neurona.run()

    # --- 2. Calcular el error y delta de salida ---
    error = salida_deseada - salida_neurona.a
    delta_salida = error * (salida_neurona.a * (1 - salida_neurona.a))

    # --- 3. Actualizar pesos y sesgo de la neurona de salida ---
    for dato in salida_neurona.inputs:
        dato.w += lr * delta_salida * dato.x
    salida_neurona.b += lr * delta_salida

    # --- 4. Retropropagación a la capa anterior ---
    if red_completa:
        ultima_capa = red_completa[-1]
        for idx, neurona in enumerate(ultima_capa):
            delta_oculta = neurona.a * (1 - neurona.a) * salida_neurona.inputs[idx].w * delta_salida
            for dato in neurona.inputs:
                dato.w += lr * delta_oculta * dato.x
            neurona.b += lr * delta_oculta

    # --- 5. Nueva propagación hacia adelante con pesos actualizados ---
    valores_actuales = [float(v) for v in entradas]
    for capa in red_completa:
        salida_capa = []
        for neurona in capa:
            for j, dato in enumerate(neurona.inputs):
                dato.x = valores_actuales[j]
            neurona.run()
            salida_capa.append(neurona.a)
        valores_actuales = salida_capa

    for j, dato in enumerate(salida_neurona.inputs):
        dato.x = valores_actuales[j]
    salida_neurona.run()

    # --- 6. Retornar salida final actualizada ---
    return float(salida_neurona.a)
