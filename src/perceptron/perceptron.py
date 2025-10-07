# ============================================================
# Politécnica de Santa Rosa
#
# Materia: Redes Neuronales
# Profesor: Jesús Salvador López Ortega
# Grupo: IRC03
# Archivo: perceptron.py
# Descripción: definición de una clase que abstrae un perceptrón
# ============================================================
import numpy as np
from perceptron.input_data import InputData
#############################################################################################################################
# Perceptrón de una sola neurona                                                                                            #
# Una neurona toma n entradas (x1, x2, ... xn), las multiplica por sus respectivos pesos (w1, w2, ...wn),                   #
# les suma un sesgo (b), y produce una salida (a) mediante una combinación lineal:                                          #
#                                                                                                                           #
# z = x1 * w1 + x2 * w2 + ... xn * wn + b                                                                                   #
# a = f(z)  # En este ejemplo, aplicamos la sigmoide como función de activación                                             #
#                                                                                                                           #
# Diagrama conceptual:                                                                                                      #
#   x1 ─┐                                                                                                                   #
#       │                                                                                                                   #
#   x2 ─┼─► [ Neurona ] ──► a                                                                                               #
#       │                                                                                                                   #
#      ...      ↑                                                                                                           #
#       │   (w1, w2, ... wn, b)                                                                                             #
#   xn ─┘                                                                                                                   #
#                                                                                                                           #
#                                                                                                                           #
# Este modelo es la base de redes más complejas. Ideal para introducir conceptos como pesos, sesgo y salida lineal.         #
#                                                                                                                           #
# NOTE: https://docs.python.org/3/tutorial/classes.html                                                                     #
#                                                                                                                           #
#############################################################################################################################

# Paso 2: Abstracción de una neurona.
#
# TODO: Define una clase "Perceptron" que contenga todos los elementos del diagrama conceptual referentes a un perceptrón.
#
# NOTE: * Considera todos los procesos que ocurren dentro de un perceptrón:
#           - La suma ponderada de las entradas
#           - La aplicación de una función de activación
#
#       * Recuerda que un perceptrón contiene:
#           - una o varias entradas de datos con un peso respectivo.
#           - un sesgo "b"
#           - una suma ponderada de las entradas "z"
#           - una salida "a" definida por su función de activación
#
class Perceptron:
    def __init__(self, inputs: list[InputData], b: float):
        self.inputs = inputs
        self.b = b
        #inicializar z y a a 0.0. Se calcularan cuando se llame a run()
        self.z = 0.0
        self.a = 0.0
        
    def forward(self):
        z_sum = 0.0 #Usar z_sum para evitar confusion con self.z 
        for input_data_obj in self.inputs: #Renombrar 'input' para evitar conflicto con la palabra clave 'input' 
            z_sum += (input_data_obj.x * input_data_obj.w)
        return z_sum + self.b
    
    def activation(self):
        #asegurarse de que self.z tenga un valor antes de usarlo 
        return 1.0 / (1.0 + np.exp(-self.z))

    def run(self):
        self.z = self.forward()        # Calcula la suma ponderada
        self.a = self.activation()     # Aplica la función de activación

    # Agregamos un método output() para obtener 'a' más limpiamente, aunque p.a también funciona.
    def output(self) -> float:
        return self.a