import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np


# --- 1. Definición del Modelo (Debe coincidir con el entrenamiento) ---
# Se necesita la definición de la clase para cargar los pesos
class FusionGRU(nn.Module):
    """
    Modelo GRU para la fusión y limpieza de sensores.
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(FusionGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU procesa la secuencia de datos (IMU, Láser, GNSS)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

        # Capa lineal para mapear la salida de la GRU a las 14 variables deseadas
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Inicializar el estado oculto (h0)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # out: (batch_size, seq_len, hidden_size)
        out, _ = self.gru(x, h0)

        # Tomar la salida del último paso de la secuencia y pasar por la capa lineal
        out = self.fc(out[:, -1, :])
        return out


# --- 2. Configuración y Carga de Recursos ---
INFERENCE_FILE = (
    "../datasets/datos_input.csv"  # Archivo con 15 columnas (Tiempo + 14 Barato)
)
OUTPUT_FILE = "../datasets/datos_limpios_modelo1_output.csv"

# 🛑🛑 CORRECCIÓN CRÍTICA DE HIPERPARÁMETROS 🛑🛑
# Deben coincidir con los valores optimizados del entrenamiento.
INPUT_SIZE = 14
OUTPUT_SIZE = 14
HIDDEN_SIZE = 128  # ¡Cambiado de 64 a 128!
NUM_LAYERS = 2

# Determinar dispositivo (Jetson Nano usará CUDA si está disponible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    # Cargar el modelo
    model = FusionGRU(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS).to(device)
    model.load_state_dict(
        torch.load("../models/fusion_gru_weights.pth", map_location=device)
    )
    model.eval()  # Modo de evaluación

    # Cargar los escaladores (CRUCIAL para la normalización/desnormalización)
    scaler_X = joblib.load("../models/scaler_X.pkl")
    scaler_Y = joblib.load("../models/scaler_Y.pkl")

    # Columnas de salida deseadas
    output_cols_names = [
        "ax_caro",
        "ay_caro",
        "az_caro",
        "wx_caro",
        "wy_caro",
        "wz_caro",
        "Lat_caro",
        "Lon_caro",
        "Alt_caro",
        "vN_caro",
        "vE_caro",
        "vU_caro",
        "d_izq_caro",
        "d_der_caro",
    ]

except FileNotFoundError:
    print("Error: No se encuentran 'fusion_gru_weights.pth' o los escaladores.")
    print("Asegúrese de ejecutar el script de entrenamiento primero.")
    exit()

# --- 3. Ejecución de la Inferencia ---

# Cargar los datos crudos
df_crudo = pd.read_csv(INFERENCE_FILE)
time_data = df_crudo["Time"]
X_crudo = df_crudo.drop("Time", axis=1).values  # 14 columnas de entrada

# Normalizar las entradas con el scaler_X entrenado
X_scaled_inf = scaler_X.transform(X_crudo)

# Convertir a tensor (añadiendo las dimensiones de Batch y Secuencia)
# El procesamiento se hace por lotes (N, 1, 14) pero el Batch Size N puede ser grande
X_tensor = torch.tensor(X_scaled_inf, dtype=torch.float32).unsqueeze(1).to(device)

# Realizar la predicción
with torch.no_grad():
    outputs_scaled = model(X_tensor)

# La salida es (N, 1, 14), la reformateamos a (N, 14)
outputs_scaled = outputs_scaled.squeeze(1).cpu().numpy()

# Desnormalizar la predicción con el scaler_Y entrenado para obtener valores reales
outputs_real = scaler_Y.inverse_transform(outputs_scaled)

# Crear el DataFrame final
df_limpio = pd.DataFrame(outputs_real, columns=output_cols_names)
df_limpio.insert(0, "Time", time_data)

# --- 4. Salida ---
df_limpio.to_csv(OUTPUT_FILE, index=False)

print(f"\n--- Inferencia Exitosa ---")
print(f"Datos limpios guardados en: {OUTPUT_FILE}")
print(
    f"El output listo para el Modelo 2 contiene {df_limpio.shape[0]} registros y {df_limpio.shape[1]} columnas."
)
print("\nPrimeros registros del output (simulación de Hardware Caro):")
print(df_limpio.head())
