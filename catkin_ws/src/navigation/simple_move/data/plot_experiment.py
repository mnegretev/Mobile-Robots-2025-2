import matplotlib.pyplot as plt
import numpy as np
import os

filename = "data.txt"
print("Ruta absoluta del archivo:", os.path.abspath(filename))

if not os.path.exists(filename):
    print("Error: El archivo no existe en esta ruta.")
else:
    print("Archivo encontrado, procediendo con la carga de datos.")


def load_experiment_data(filename):
    experiments = []
    current_experiment = {"params": {}, "data": []}  # Se inicializa correctamente

    with open(filename, "r") as file:
        for line in file:
            values = list(map(float, line.strip().split(",")))

            if len(values) == 7:  # Asegurar que la línea tiene el formato esperado
                current_experiment["data"].append(values)
            else:
                print(f"Ignorando línea mal formada: {line.strip()}")  # Depuración

    if current_experiment["data"]:  # Solo añadir si hay datos
        experiments.append(current_experiment)

    return experiments if experiments else None  # Evitar retornar None


def plot_experiment(experiment):
    data = np.array(experiment["data"])
    
    if data.shape[0] == 0:
        return
    
    robot_x, robot_y = data[:, 0], data[:, 1]
    goal_x, goal_y = data[:, 3], data[:, 4]
    v, w = data[:, 5], data[:, 6]
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    
    # Graficar trayectorias
    axs[0].plot(goal_x, goal_y, "ro-", label="Ruta Deseada")
    axs[0].plot(robot_x, robot_y, "bo-", label="Ruta Seguida")
    axs[0].set_title("Comparación entre Ruta Deseada y Ruta Seguida")
    axs[0].set_xlabel("Posición X")
    axs[0].set_ylabel("Posición Y")
    axs[0].legend()
    axs[0].grid()
    
    # Graficar velocidades
    axs[1].plot(v, label="Velocidad Lineal (v)")
    axs[1].plot(w, label="Velocidad Angular (w)")
    axs[1].set_title("Velocidades del Robot")
    axs[1].set_xlabel("Tiempo")
    axs[1].set_ylabel("Velocidad")
    axs[1].legend()
    axs[1].grid()
    
    plt.suptitle(experiment["params"])
    plt.show()

if __name__ == "__main__":
    filename = "data.txt"
    experiments = load_experiment_data(filename)
    
    for exp in experiments:
        plot_experiment(exp)
