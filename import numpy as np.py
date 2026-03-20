import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import random
import time

"""
Configuracion

"""
Num_paquetes = 100
Tamano_paquete = 1024 #bytes
velocidad_red = 100000 #bytes por segundo

"""
Listas para almacenar datos

"""
latencias = []
paquetes_envidas = []
paquetes_recibidos = []
perdidos = 0

print("Simulando tráfico de datos . . .\n")

for i in range(Num_paquetes):
    tiempo_envio = time.time()

    """
    Simular latencia (entre 10ns y 100ns)

    """
    latencia = random.uniform(0.01,0.1)
    time.sleep(latencia)

    """
    Simular perdidas de paquetes (10%)
    """
    if random.random () < 0.1:
        perdidos += 1
        continue

    tiempo_recepcion = time.time()
    latencia.append(tiempo_recepcion - tiempo_envio)
    paquetes_envidas.append(Tamano_paquete)
    paquetes_recibidos.append(Tamano_paquete)

    """
    Metricas
    """
total_enviados = len(paquetes_envidas)
total_recibidos = len(paquetes_recibidos)
tasa_perdida = perdidos / Num_paquetes

latencia_promedio = np.mean(latencias)

throughput = (sum(paquetes_recibidos)/sum (latencia)) if latencias else 0