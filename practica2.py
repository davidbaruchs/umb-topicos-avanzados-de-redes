"""
Sistemas de Tráfico de Datos con IA
"""

import numpy as np
import pandas as pd 
import random
import time
import matplotlib.pyplot as plt
from  sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

"""
Generar Dataset
"""

def generar_datos(n=1000):
    np.random.seed(42)
    
    data = pd.DataFrame({
        "paquetes": np.random.randint(100, 5000, n),
        "bytes": np.random.randint(1000, 60000, n),
        "duracion": np.random.uniform(0.1, 15, n),
        "protocolo": np.random.choice([0,1], n),
    })

    condiciones = [
        (data["bytes"] > 45000),
        (data["paquetes"] < 2000),
    ]
    
    opciones = ["ataques", "videos"]
    
    data["tipo"] = np.select(condiciones, opciones, default="normal")

    return data

"""
Entrenamiento del Modelo
"""

def entrenar_modelo(data):
    X = data[["paquetes", "bytes", "duracion", "protocolo"]]
    y = data["tipo"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = RandomForestClassifier(n_estimators=100)
    modelo.fit(X_train, y_train)

    predicciones = modelo.predict(X_test)

    print("\nEvaluación del Modelo:")
    print("Accuracy:", accuracy_score(y_test, predicciones))
    print("Reporte de Clasificación:\n", classification_report(y_test, predicciones))

    return modelo

"""
Simulación en Tiempo Real
"""

def simulacion_tiempo_real(modelo, iteraciones=10):
    for i in range(iteraciones):
        paquetes = random.randint(100, 5000)
        bytes = random.randint(1000, 60000)
        duracion = random.uniform(0.1, 15)
        protocolo = random.choice([0,1])

        # CORRECCIÓN: array 2D
        muestra = np.array([[paquetes, bytes, duracion, protocolo]])
        pred = modelo.predict(muestra)[0]

        print(f"Iteracion {i+1}")
        print(f"Paquetes: {paquetes}, Bytes: {bytes}, Duracion: {duracion:.2f}, Protocolo: {protocolo}")
        print(f"Clasificacion: {pred}")
        print("-" * 50)

        time.sleep(1)

"""
Graficas
"""

def graficas(data):
    plt.figure()
    plt.hist(data["bytes"])
    plt.title("Distribucion de Bytes")
    plt.xlabel("Bytes")
    plt.ylabel("Frecuencia")
    plt.grid()
    plt.show()

    plt.figure()
    plt.hist(data["paquetes"])
    plt.title("Distribucion de Paquetes")
    plt.xlabel("Paquetes")
    plt.ylabel("Frecuencia")
    plt.grid()
    plt.show()


    plt.figure()
    data["tipo"].value_counts().plot(kind="bar")
    plt.title("Tipos de Trafico")
    plt.xlabel("Tipo")
    plt.ylabel("Cantidad")
    plt.grid()
    plt.show()

"""
Main
"""
if __name__ == "__main__":
    data = generar_datos(n=1000)
    print("Dataset Generado: ")
    print(data.head())

    modelo = entrenar_modelo(data)

    graficas(data)

    simulacion_tiempo_real(modelo, iteraciones=10) 