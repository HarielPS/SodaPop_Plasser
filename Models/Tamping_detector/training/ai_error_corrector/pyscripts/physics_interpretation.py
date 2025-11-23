import pandas as pd
import numpy as np

# --- 1. Definición de Constantes de Ingeniería ---
# Ancho de la vía (trocha estándar) en mm
W = 1435
# Elevación ideal de la vía (asumida para el cálculo del Hundimiento) en metros.
Y_IDEAL_M = 2240.35
# Precisión milimétrica requerida (0.1 mm)

# --- 2. Carga y Selección de Datos Limpios (Output del Modelo 1) ---
FILE_PATH = "../datasets/datos_limpios_modelo1_output.csv"
data = pd.read_csv(FILE_PATH)

# Renombrar columnas para facilitar el manejo (Asumiendo que el output ya es el limpio "_caro")
df_clean = data.copy()
df_clean.columns = [
    "Time",
    "ax",
    "ay",
    "az",
    "wx",
    "wy",
    "wz",
    "Lat",
    "Lon",
    "Alt",
    "vN",
    "vE",
    "vU",
    "d_izq",
    "d_der",
]

# --- 3. Inicialización del DataFrame de Salida (Input del Modelo 2) ---
df_estado = pd.DataFrame(df_clean["Time"])

# --- 4. Transformaciones Físicas y Geométricas ---

# A. POSICIÓN ABSOLUTA & VELOCIDAD (Sin cambios, CM y Alta Precisión)
df_estado["Y_pos_m"] = df_clean["Alt"]
df_estado["Z_pos_m"] = np.cumsum(df_clean["vN"] * df_clean["Time"].diff().fillna(0))
df_estado["Z_pos_m"] = df_estado["Z_pos_m"] - df_estado["Z_pos_m"].iloc[0]
df_estado["X_pos_Lat"] = df_clean["Lat"]
df_estado["X_pos_Lon"] = df_clean["Lon"]
df_estado["vx_ms"] = df_clean["vE"]
df_estado["vy_ms"] = df_clean["vU"]
df_estado["vz_ms"] = df_clean["vN"]

# B. ORIENTACIÓN (Roll, Pitch, Yaw)
roll_rad = np.arctan((df_clean["d_der"] - df_clean["d_izq"]) / W)
df_estado["Roll_rad"] = roll_rad
df_estado["Pitch_rad"] = df_clean["wy"]
df_estado["Yaw_rad"] = df_clean["wz"]

# C. DESVIACIONES MILIMÉTRICAS (VARIABLES CLAVE PARA LA RECETA)

# 1. Hundimiento de Riel (Vertical) [mm]
# El GNSS (Alt) nos da la altura del centro del sensor.
# La desviación vertical es (Y_IDEAL - Alt_sensor) * 1000.

# Hundimiento Riel Izquierdo [mm]
# La distancia láser d_izq es la distancia del sensor al riel.
# Asumimos que la altura ideal del riel es (Alt_sensor_ideal - d_ideal_izq)
# El hundimiento es la diferencia de la altura actual del riel respecto a la altura ideal.
df_estado["Hundimiento_Izq_mm"] = (Y_IDEAL_M - df_clean["Alt"]) * 1000 - (
    df_clean["d_izq"] - 500.0
)

# Hundimiento Riel Derecho [mm]
df_estado["Hundimiento_Der_mm"] = (Y_IDEAL_M - df_clean["Alt"]) * 1000 - (
    df_clean["d_der"] - 500.0
)

# 2. Desviación Lateral (Alineación) [mm]
# Estas variables indican el error horizontal de la vía.
# Se usan los valores de aceleración lateral (ax) y angular (wx, wy) como proxy del error de alineación lateral.

# Alineación Riel Izquierdo [mm]
# Usamos ax (aceleración lateral) como proxy de la fuerza de desvío
df_estado["Alineacion_Izq_mm"] = df_clean["ax"] * 500

# Alineación Riel Derecho [mm]
# Simulación de que la alineación lateral se relaciona con la fuerza lateral y la orientación.
df_estado["Alineacion_Der_mm"] = -df_clean["ax"] * 500


# --- 5. Resultados y Guardar ---

# El input para el Modelo 2 ahora tiene 4 variables de defecto
df_final_input_modelo2 = df_estado[
    [
        "Time",
        "X_pos_Lat",
        "X_pos_Lon",
        "Y_pos_m",
        "Z_pos_m",
        "vx_ms",
        "vy_ms",
        "vz_ms",
        "Roll_rad",
        "Pitch_rad",
        "Yaw_rad",
        "Hundimiento_Izq_mm",
        "Hundimiento_Der_mm",
        "Alineacion_Izq_mm",
        "Alineacion_Der_mm",
    ]
].copy()

# Guardar el dataset final (Input para el Modelo 2)
df_final_input_modelo2.to_csv(
    "../datasets/input_modelo2_variables_estado.csv", index=False
)

print(
    "Dataset de Variables de Estado MEJORADO (Input para Modelo 2) generado con éxito."
)
print(
    "\nVariables de Salida (Input para Modelo 2) -- ¡Ahora con 4 Defectos Separados!:"
)
print(df_final_input_modelo2.head())
