
#!/usr/bin/env python3

import time
import numpy as np
import random
import pandas as pd
from itertools import product
from nn_training import NeuralNetwork, load_dataset

# Parametrización de combinaciones
learning_rates = [0.5, 1.0, 3.0, 10.0]
epochs_list = [3, 10, 50, 100]
batch_sizes = [5, 10, 30, 100]
combinaciones = list(product(epochs_list, batch_sizes, learning_rates))

# Cargar el dataset (ajustar ruta si es necesario)
dataset_folder = "../handwritten_digits/"
training_dataset, testing_dataset = load_dataset(dataset_folder)

# Función de evaluación
def evaluar_red(nn, testing_dataset, cantidad=100):
    correctos = 0
    for _ in range(cantidad):
        img, label = random.choice(testing_dataset)
        y = nn.feedforward(img).transpose()
        expected = np.argmax(label)
        predicted = np.argmax(y)
        if expected == predicted:
            correctos += 1
    return correctos

# DataFrame para guardar resultados
resultados = []

for epochs, batch_size, learning_rate in combinaciones:
    print(f"Entrenando con epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}")
    nn = NeuralNetwork([784, 30, 10])  # Arquitectura fija sugerida
    start_time = time.time()
    nn.train_by_SGD(training_dataset, epochs, batch_size, learning_rate)
    end_time = time.time()
    tiempo = end_time - start_time

    aciertos = evaluar_red(nn, testing_dataset, cantidad=100)
    precision = aciertos / 100 * 100

    resultados.append({
        "Epochs": epochs,
        "Batch size": batch_size,
        "Learning rate": learning_rate,
        "Tiempo (s)": round(tiempo, 2),
        "Aciertos (de 100)": aciertos,
        "% Acierto": round(precision, 2)
    })

# Guardar en Excel
df = pd.DataFrame(resultados)
df.to_excel("resultados_64_pruebas.xlsx", index=False)
print("\n✅ Evaluación completa. Resultados guardados en resultados_64_pruebas.xlsx")
