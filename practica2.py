"""
Sistemas de trafico de datos con ia

"""

import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


"""
generar datseet
"""

def generar_datos(n=1000):
    np.random.seed(42)

    data = pd.DataFrame({
        "paquetes": np.random
        "bytes": np.random.randit(1000, 60000, n),
        "duracion":np.random.uniforme(0.1,15, n),
        "protocolo": np.random.choice([0,1],1),
    })


    condiciones = [
        (data["bytes"]>45000),
        (data["parquetes"]<2000),


    ]  

    opciones =["ataque","video"]

    data["tipo"] = np.select(condiciones, opciones, default="normal")

    return data
