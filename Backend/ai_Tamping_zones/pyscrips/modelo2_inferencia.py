import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ===========================
# 1. Rutas
# ===========================
# CSV NUEVO que quieres probar (excel verdadero)
# CSV NUEVO que quieres probar (excel verdadero)
FILE_STATE_NEW = "../datasets/estado_nueva_medicion.csv"

# Modelo entrenado
MODEL_PATH = "../models/modelo2_tamping.pt"

# Salidas
OUTPUT_CSV  = "../results/estado_nueva_medicion_con_pred.csv"
OUTPUT_JSON = "../results/estado_nueva_medicion_sections.json"

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)


# ===========================
# 2. Definir misma arquitectura que en entrenamiento
# ===========================
class MLPReg(nn.Module):
    def __init__(self, in_dim, hidden=[64, 32], out_dim=2):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ===========================
# 3. Cargar modelo
# ===========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo para inferencia:", device)

# mismo orden de features que en el script de entrenamiento
feature_cols = [
    "Hundimiento_mm",
    "Alineacion_mm",
    "Roll_rad",
    "Pitch_rad",
    "Yaw_rad",
    "vx_ms",
]

in_dim = len(feature_cols)
model = MLPReg(in_dim=in_dim, hidden=[64, 32], out_dim=2).to(device)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()
print(f"Modelo cargado desde: {MODEL_PATH}")


# ===========================
# 4. Cargar CSV nuevo
# ===========================
df_new = pd.read_csv(FILE_STATE_NEW)
print("Columnas del CSV nuevo:", df_new.columns.tolist())

# Validar columnas necesarias
needed = [
    "Time",
    "X_pos_Lat",
    "X_pos_Lon",
    "Y_pos_m",
    "Hundimiento_mm",
    "Alineacion_mm",
    "Roll_rad",
    "Pitch_rad",
    "Yaw_rad",
    "vx_ms",
]
for col in needed:
    if col not in df_new.columns:
        raise ValueError(f"Falta la columna '{col}' en {FILE_STATE_NEW}")

# ===========================
# 5. Construir X para predicción
# ===========================
X = df_new[feature_cols].astype(np.float32).values
X_t = torch.from_numpy(X).to(device)

with torch.no_grad():
    preds = model(X_t).cpu().numpy()

ajuste_izq_pred = preds[:, 0]
ajuste_der_pred = preds[:, 1]

# Clip por seguridad (no negativos)
ajuste_izq_pred = np.clip(ajuste_izq_pred, 0.0, 200.0)
ajuste_der_pred = np.clip(ajuste_der_pred, 0.0, 200.0)

# Agregar columnas al dataframe
df_new["ajuste_izquierdo_pred_mm"] = ajuste_izq_pred
df_new["ajuste_derecho_pred_mm"]   = ajuste_der_pred

# Guardar CSV con predicciones por fila
df_new.to_csv(OUTPUT_CSV, index=False)
print(f"CSV con predicciones guardado en: {OUTPUT_CSV}")


# ===========================
# 6. Generar secciones de 40 m para la tamping
# ===========================
L_SECCION = 40.0  # largo de la máquina en metros

dist  = df_new["Y_pos_m"].astype(float).values
lat   = df_new["X_pos_Lat"].astype(float).values
lon   = df_new["X_pos_Lon"].astype(float).values
hund  = df_new["Hundimiento_mm"].astype(float).values
alin  = df_new["Alineacion_mm"].astype(float).values
roll  = df_new["Roll_rad"].astype(float).values
pitch = df_new["Pitch_rad"].astype(float).values
yaw   = df_new["Yaw_rad"].astype(float).values
speed = df_new["vx_ms"].astype(float).values

start_track = dist.min()
end_track   = dist.max()

secciones_output = []
sec_id = 0
current_start = start_track

while current_start < end_track:
    current_end = min(current_start + L_SECCION, end_track)
    mask = (dist >= current_start) & (dist <= current_end)
    if not np.any(mask):
        current_start = current_end
        continue

    lat_inicio = float(lat[mask][0])
    lat_fin    = float(lat[mask][-1])
    lon_inicio = float(lon[mask][0])
    lon_fin    = float(lon[mask][-1])

    hund_prom  = max(0.0, float(hund[mask].mean()))
    alin_prom  = float(alin[mask].mean())
    roll_mean  = float(roll[mask].mean())
    pitch_mean = float(pitch[mask].mean())
    yaw_mean   = float(yaw[mask].mean())
    speed_mean = float(speed[mask].mean())

    # Build features para la sección
    x_sec = np.array(
        [hund_prom, alin_prom, roll_mean, pitch_mean, yaw_mean, speed_mean],
        dtype=np.float32
    )

    with torch.no_grad():
        x_t_sec = torch.from_numpy(x_sec).unsqueeze(0).to(device)
        pred_sec = model(x_t_sec).cpu().numpy()[0]

    ajuste_izq_sec = float(np.clip(pred_sec[0], 0.0, 200.0))
    ajuste_der_sec = float(np.clip(pred_sec[1], 0.0, 200.0))

    secciones_output.append({
        "seccion_id": int(sec_id),
        "lat_inicio": round(lat_inicio, 8),
        "lat_fin": round(lat_fin, 8),
        "lon_inicio": round(lon_inicio, 8),
        "lon_fin": round(lon_fin, 8),
        "hundimiento_mm": round(hund_prom, 4),
        "ajuste_izquierdo_mm": round(ajuste_izq_sec, 4),
        "ajuste_derecho_mm": round(ajuste_der_sec, 4)
    })

    sec_id += 1
    current_start = current_end

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(secciones_output, f, indent=2, ensure_ascii=False)

print(f"JSON de secciones guardado en: {OUTPUT_JSON}")
print("Primeras secciones:")
for sec in secciones_output[:3]:
    print(sec)
