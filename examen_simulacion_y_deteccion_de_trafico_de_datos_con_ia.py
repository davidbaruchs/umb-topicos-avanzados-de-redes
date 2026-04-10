"""
EXAMEN: SIMULACIÓN Y DETECCIÓN DE TRÁFICO DE DATOS CON IA
"""

import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


"""
Generación de datos (Simulación)
"""
def generar_datos(n=200):
    np.random.seed(42)

    data = pd.DataFrame({
        "paquetes": np.random.randint(100, 5000, n),
        "tamano": np.random.randint(500, 50000, n),
        "tiempo": np.random.uniform(0.1, 10, n),
        "latencia": np.random.uniform(1, 150, n),
        "ancho_banda": np.random.uniform(1, 100, n)
    })

    data["tipo"] = (
        (data["latencia"] > 100) |
        (data["tamano"] > 40000) |
        (data["paquetes"] > 4000)
    ).astype(int)

    return data


"""
ENTRENAMIENTO DEL MODELO
"""
def entrenar_modelo(data):
    X = data[["paquetes", "tamano", "tiempo", "latencia", "ancho_banda"]]
    y = data["tipo"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = RandomForestClassifier(n_estimators=100)
    modelo.fit(X_train, y_train)

    pred = modelo.predict(X_test)

    print("Evaluacion del modelo")
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, pred))

    return modelo


"""
SIMULACION EN TIEMPO REAL
"""
def simulacion_tiempo_real(modelo, iteraciones=30):

    plt.ion()

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()

    x_data = []
    paquetes_data = []
    clasificacion_data = []

    print("\nSimulacion en tiempo real:\n")

    for i in range(iteraciones):

        paquetes = random.randint(100, 5000)
        tamano = random.randint(500, 50000)
        tiempo_dato = random.uniform(0.1, 10)
        latencia = random.uniform(1, 150)
        ancho_banda = random.uniform(1, 100)

        muestra = np.array([[paquetes, tamano, tiempo_dato, latencia, ancho_banda]])
        pred = modelo.predict(muestra)[0]

        print("Iteracion:", i + 1)
        print("Paquetes:", paquetes, "Tamano:", tamano, "Latencia:", latencia)
        print("Clasificacion:", "ANOMALO" if pred == 1 else "NORMAL")
        print("-" * 40)

        x_data.append(i)
        paquetes_data.append(paquetes)
        clasificacion_data.append(pred)

       
        ax1.clear()
        ax1.plot(x_data, paquetes_data)
        ax1.set_title("Tráfico vs Tiempo")
        ax1.set_xlabel("Tiempo")
        ax1.set_ylabel("Paquetes")

        
        ax2.clear()
        ax2.bar(range(len(paquetes_data)), paquetes_data)
        ax2.set_title("Paquetes Enviados")
        ax2.set_xlabel("Iteración")
        ax2.set_ylabel("Paquetes")

        
        ax3.clear()

        normales_x = [x_data[j] for j in range(len(clasificacion_data)) if clasificacion_data[j] == 0]
        anom_x = [x_data[j] for j in range(len(clasificacion_data)) if clasificacion_data[j] == 1]

        normales_y = [paquetes_data[j] for j in range(len(clasificacion_data)) if clasificacion_data[j] == 0]
        anom_y = [paquetes_data[j] for j in range(len(clasificacion_data)) if clasificacion_data[j] == 1]

        ax3.scatter(normales_x, normales_y, label="Normal")
        ax3.scatter(anom_x, anom_y, label="Anomalo")

        ax3.set_title("Clasificación del tráfico")
        ax3.set_xlabel("Tiempo")
        ax3.set_ylabel("Paquetes")
        ax3.legend()

        plt.pause(0.5)
        time.sleep(0.5)

    plt.ioff()
    plt.show()


"""
GRAFICAS
"""
def graficas(data):

    from mpl_toolkits.mplot3d import Axes3D

    plt.ion()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x_data = []
    y_data = []
    z_data = []

    print("\nGraficas en tiempo real:\n")

    for i in range(len(data)):

        x_data.append(data["paquetes"].iloc[i])
        y_data.append(data["tamano"].iloc[i])
        z_data.append(data["latencia"].iloc[i])

        ax.clear()

        ax.scatter(x_data, y_data, z_data)

        ax.set_xlabel("Paquetes")
        ax.set_ylabel("Tamano")
        ax.set_zlabel("Latencia")
        ax.set_title("Comportamiento del Trafico de Red")

        plt.pause(0.05)

    plt.ioff()
    plt.show()


"""
MAIN
"""
if __name__ == "__main__":

    data = generar_datos(200)

    print("Dataset generado:")
    print(data.head())

    modelo = entrenar_modelo(data)

    graficas(data)

    simulacion_tiempo_real(modelo)