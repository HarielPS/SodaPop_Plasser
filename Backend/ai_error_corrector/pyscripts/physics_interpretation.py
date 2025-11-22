import pandas as pd
import numpy as np

# --- 1. Definición de Constantes de Ingeniería ---
# Ancho de la vía (trocha estándar) en mm
W = 1435
# Elevación ideal de la vía (asumida para el cálculo del Hundimiento) en metros
Y_IDEAL_M = 2240.35
# Factor de conversión para Lat/Lon a distancia aproximada en metros (cerca del ecuador)
# 1 grado de Latitud es ~111.1 km; usaremos este factor para simplificar la proyección.
LAT_TO_M = 111100.0

# --- 2. Carga y Selección de Datos Limpios (Output del Modelo 1) ---
FILE_PATH = "../datasets/datos_limpios_modelo1_output.csv"
data = pd.read_csv(FILE_PATH)

# Seleccionar solo las columnas de "Hardware Caro" (Output limpio del Modelo 1)
clean_cols = [col for col in data.columns if "caro" in col]
df_clean = data[["Time"] + clean_cols].copy()

# Renombrar columnas para facilitar el manejo
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
# Aquí creamos las 10 variables de estado y las 2 variables de desviación milimétrica
df_estado = pd.DataFrame(df_clean["Time"])

# --- 4. Transformaciones Físicas y Geométricas ---

# A. POSICIÓN ABSOLUTA (X_pos, Y_pos, Z_pos) - Precisión CM
# Y_pos: Altitud (directamente de GNSS).
df_estado["Y_pos_m"] = df_clean["Alt"]

# Z_pos (Longitudinal / KM): Cálculo de la distancia recorrida (simplificado)
# Asumimos que la mayor parte del movimiento es Norte (Lat).
df_estado["Z_pos_m"] = np.cumsum(df_clean["vN"] * df_clean["Time"].diff().fillna(0))
df_estado["Z_pos_m"] = (
    df_estado["Z_pos_m"] - df_estado["Z_pos_m"].iloc[0]
)  # Normalizar inicio en 0

# X_pos: Asumimos que Lat/Lon ya está en el centro de la vía.
# Para un cálculo real, esto requeriría una conversión UTM precisa.
# Aquí solo usamos Lat/Lon como referencia centimétrica.
df_estado["X_pos_Lat"] = df_clean["Lat"]
df_estado["X_pos_Lon"] = df_clean["Lon"]

# B. VELOCIDAD (vx, vy, vz) - Alta Precisión
# vx (lateral), vy (vertical), vz (longitudinal)
# Asumimos vN es longitudinal (vz), vE es lateral (vx), vU es vertical (vy) en el marco de la vía
df_estado["vx_ms"] = df_clean["vE"]  # Velocidad lateral
df_estado["vy_ms"] = df_clean["vU"]  # Velocidad vertical
df_estado["vz_ms"] = df_clean["vN"]  # Velocidad longitudinal

# C. ORIENTACIÓN (Roll, Pitch, Yaw) - Precisión Angular (Milimétrica)
# Roll (phi): Alabeo / Torsión (Clave del desnivel)
# Usando la fórmula trigonométrica simplificada del láser:
roll_rad = np.arctan((df_clean["d_der"] - df_clean["d_izq"]) / W)
df_estado["Roll_rad"] = roll_rad

# Pitch (theta): Cabeceo / Hundimiento longitudinal (Clave de la rampa)
# Simplificado: se correlaciona fuertemente con la velocidad angular wy.
df_estado["Pitch_rad"] = df_clean["wy"]

# Yaw (psi): Guiñada / Curvatura horizontal
# Simplificado: se correlaciona con la velocidad angular wz.
df_estado["Yaw_rad"] = df_clean["wz"]

# D. DESVIACIONES MILIMÉTRICAS (El Lift) - Precisión MM

# Desviación Vertical (Hundimiento) [mm]
# El hundimiento es la diferencia entre dónde debería estar (Y_IDEAL) y dónde está (Y_pos).
# Multiplicamos por 1000 para obtener el valor en milímetros (mm).
df_estado["Hundimiento_mm"] = (Y_IDEAL_M - df_estado["Y_pos_m"]) * 1000

# Desviación Lateral (Alineación) [mm]
# Se calcula la desviación del eje central basado en la aceleración lateral (ax)
# El movimiento lateral indica que la vía no sigue una línea recta/curva ideal.
# Se usa la aceleración lateral como proxy del error de alineación (transformada).
# Se asume que el valor es el error de alineación del carril (Alineación lateral).
# Nota: La alineación real requiere integración compleja de ax y wx. Aquí usamos un factor simple.
df_estado["Alineacion_mm"] = (
    df_clean["ax"] * 1000
)  # Usamos ax como proxy de la fuerza lateral

# --- 5. Resultados y Guardar ---

# Limpiar columnas intermedias si es necesario y reordenar para el Modelo 2
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
        "Hundimiento_mm",
        "Alineacion_mm",
    ]
].copy()

# Guardar el dataset final (Input para el Modelo 2)
df_final_input_modelo2.to_csv(
    "../datasets/input_modelo2_variables_estado.csv", index=False
)

print("Dataset de Variables de Estado generado con éxito.")
print("Archivo guardado como 'input_modelo2_variables_estado.csv'")
print("\nVariables de Salida (Input para Modelo 2):")
print(df_final_input_modelo2.head())
