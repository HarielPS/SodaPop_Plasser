import pandas as pd
import numpy as np

# ==============================================================
# CONFIGURACIÓN
# ==============================================================
FILE_INPUT = "../datasets/input_modelo2_variables_estado.csv"
FILE_OUTPUT = "../datasets/input_modelo2_variables_estado_ampliado.csv"
# Longitud de la extensión que queremos (en metros)
LENGTH_TO_ADD = 1000.0
# Número de puntos que usaremos del dataset original como "plantilla"
N_REPLICATE = 100

# Parámetros para la simulación de Hundimiento/Alineación
HUND_MAX_AMPLITUDE = 10.0  # Máxima variación simulada de 10 mm
WAVE_FREQUENCY = 0.1  # Frecuencia para simular ondulaciones de la vía

# ==============================================================
# 1. CARGAR Y PREPARAR DATOS
# ==============================================================
try:
    df = pd.read_csv(FILE_INPUT)
except FileNotFoundError:
    print(
        f"ERROR: No se encontró el archivo {FILE_INPUT}. Asegúrese de que esté en el directorio correcto."
    )
    exit()

print(f"Dataset original cargado. Longitud inicial: {len(df)} filas.")

# Ordenar por distancia (Y_pos_m) y resetear el índice
df = df.sort_values(by="Y_pos_m").reset_index(drop=True)

# Obtener la última fila como punto de partida
last_row = df.iloc[-1]
start_y = last_row["Y_pos_m"]
end_y = start_y + LENGTH_TO_ADD

# Calcular el paso promedio de cada columna para las últimas N_REPLICATE filas
df_trend = df.tail(N_REPLICATE)
Y_step_avg = df_trend["Y_pos_m"].diff().mean()
Time_step_avg = df_trend["Time"].diff().mean()
Lat_step_avg = df_trend["X_pos_Lat"].diff().mean()
Lon_step_avg = df_trend["X_pos_Lon"].diff().mean()

# Si el paso promedio es cero o nulo, usar un valor por defecto (asumimos 10cm por fila)
if pd.isna(Y_step_avg) or Y_step_avg <= 0:
    Y_step_avg = 0.1

# ==============================================================
# 2. GENERAR NUEVOS PUNTOS DE DISTANCIA
# ==============================================================
# Calcular cuántos nuevos puntos necesitamos
num_new_points = int(LENGTH_TO_ADD / Y_step_avg)
print(f"Se generarán aproximadamente {num_new_points} nuevos puntos.")

# Generar la secuencia de distancia (Y_pos_m) para la extensión
new_y_pos = np.linspace(start_y + Y_step_avg, end_y, num_new_points)

# Crear el DataFrame para la extensión
df_extension = pd.DataFrame()
df_extension["Y_pos_m"] = new_y_pos

# ==============================================================
# 3. INTERPOLAR Y SIMULAR EL RESTO DE LAS COLUMNAS
# ==============================================================

# Inicializar valores base para la extensión desde la última fila original
current_lat = last_row["X_pos_Lat"]
current_lon = last_row["X_pos_Lon"]
current_time = last_row["Time"]
current_z = last_row["Z_pos_m"]

# Columnas que se replicarán con valores promedio o ruido
cols_to_replicate = [
    "Z_pos_m",
    "vx_ms",
    "vy_ms",
    "vz_ms",
    "Roll_rad",
    "Pitch_rad",
    "Yaw_rad",
]
# Calcular los promedios de estas columnas en las últimas filas
avg_values = df_trend[cols_to_replicate].mean()


new_data = []
for i, y_pos in enumerate(new_y_pos):
    # 3.1. Progresión de Latitud/Longitud y Tiempo
    current_lat += Lat_step_avg
    current_lon += Lon_step_avg
    current_time += Time_step_avg

    # 3.2. Simulación de Hundimiento/Alineación (con función sinusoidal)
    # Esto es crucial para darle nueva información al modelo
    wave_input = y_pos * WAVE_FREQUENCY

    # Base del hundimiento:
    hund_base = HUND_MAX_AMPLITUDE * (0.5 + 0.5 * np.sin(wave_input))
    # Alineación: Mismo patrón pero centrado en cero, con un pequeño desplazamiento
    alin_base = (HUND_MAX_AMPLITUDE / 3) * np.cos(wave_input + np.pi / 4)

    # Aplicar sesgo de riel para simular la diferencia
    hund_izq = hund_base * (0.95 + 0.1 * np.random.rand())
    hund_der = hund_base * (1.05 - 0.1 * np.random.rand())
    alin_izq = alin_base * (0.9 + 0.2 * np.random.rand())
    alin_der = -alin_base * (0.9 + 0.2 * np.random.rand())

    # 3.3. Construir la fila
    row = {
        "Time": current_time,
        "X_pos_Lat": current_lat,
        "X_pos_Lon": current_lon,
        "Y_pos_m": y_pos,
        # Columnas replicadas con valores promedio (o ruido leve)
        "Z_pos_m": avg_values["Z_pos_m"] + np.random.normal(0, 0.01),
        "vx_ms": avg_values["vx_ms"] + np.random.normal(0, 0.005),
        "vy_ms": avg_values["vy_ms"],
        "vz_ms": avg_values["vz_ms"],
        "Roll_rad": avg_values["Roll_rad"] + np.random.normal(0, 0.001),
        "Pitch_rad": avg_values["Pitch_rad"] + np.random.normal(0, 0.001),
        "Yaw_rad": avg_values["Yaw_rad"] + np.random.normal(0, 0.001),
        # Valores simulados
        "Hundimiento_Izq_mm": hund_izq,
        "Hundimiento_Der_mm": hund_der,
        "Alineacion_Izq_mm": alin_izq,
        "Alineacion_Der_mm": alin_der,
    }
    new_data.append(row)

df_extension = pd.DataFrame(new_data)

# ==============================================================
# 4. CONCATENAR Y GUARDAR
# ==============================================================
df_final = pd.concat([df, df_extension], ignore_index=True)

# Redondear todas las columnas a 6 decimales para un archivo limpio
df_final = df_final.round(6)

# Guardar el nuevo dataset
df_final.to_csv(FILE_OUTPUT, index=False)

print("\n---------------------------------------------------------")
print(f"✔ ¡ÉXITO! Dataset ampliado generado en: {FILE_OUTPUT}")
print(f"Longitud total del nuevo dataset: {len(df_final)} filas.")
print(
    f"Distancia total de la vía (Y_pos_m): {df_final['Y_pos_m'].max() - df_final['Y_pos_m'].min():.2f} metros."
)
print("---------------------------------------------------------")
