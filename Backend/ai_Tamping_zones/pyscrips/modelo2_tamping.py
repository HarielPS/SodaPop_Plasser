import json
import math
import time
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim


# ===========================
# 1. Cargar datasets
# ===========================
# Desde pyscrips subir a ai_Tamping_zones/
FILE_STATE = "../datasets/plantilla_etiquetas_modelo2.csv"

# Para ai_error_corrector subir 2 niveles hacia Backend/
FILE_RAW   = "../../ai_error_corrector/datasets/datos_inspeccion_vias.csv"

df_state = pd.read_csv(FILE_STATE)
df_raw   = pd.read_csv(FILE_RAW)

print("Columnas estado:", df_state.columns.tolist())
print("Columnas raw   :", df_raw.columns.tolist())


# ===========================
# 2. Mapeo de columnas
# ===========================
TIME_COL   = "Time"
LAT_COL    = "X_pos_Lat"
LON_COL    = "X_pos_Lon"
M_COL      = "Y_pos_m"
HUND_COL   = "Hundimiento_mm"
ALIN_COL   = "Alineacion_mm"
ROLL_COL   = "Roll_rad"
PITCH_COL  = "Pitch_rad"
YAW_COL    = "Yaw_rad"
SPEED_COL  = "vx_ms"

# etiquetas que TÚ debes agregar al CSV
LEFT_COL   = "ajuste_izquierdo_mm"
RIGHT_COL  = "ajuste_derecho_mm"

for col in [M_COL, LAT_COL, LON_COL, HUND_COL, ALIN_COL,
            ROLL_COL, PITCH_COL, YAW_COL, SPEED_COL,
            LEFT_COL, RIGHT_COL]:
    if col not in df_state.columns:
        raise ValueError(f"Falta la columna '{col}' en input_modelo2_variables_estado.csv")

# ordenar por posición
df_state = df_state.sort_values(M_COL).reset_index(drop=True)

# =====================================================
# GENERAR ETIQUETAS AUTOMÁTICAS SI ESTÁN VACÍAS
# =====================================================

# Convertir columnas de etiquetas a numérico (o NaN si vacías)
df_state[LEFT_COL]  = pd.to_numeric(df_state[LEFT_COL],  errors="coerce")
df_state[RIGHT_COL] = pd.to_numeric(df_state[RIGHT_COL], errors="coerce")

# Detectar si todas están vacías
all_empty = df_state[[LEFT_COL, RIGHT_COL]].isna().all().all()

if all_empty:
    print("⚠ No hay etiquetas reales. Se generarán etiquetas automáticas...")

    hund = df_state[HUND_COL].astype(float).clip(lower=0)
    alin = df_state[ALIN_COL].astype(float)

    # regla ingenieril:
    # · la mitad del hundimiento es la base del levantamiento
    # · la alineación genera asimetría IZQ-DER
    # · suavizado para estabilidad
    base = hund * 0.5
    bias = alin / (abs(alin) + 1e-6)  # -1, 0, 1

    ajuste_izq = (base * (1.0 - 0.2*bias)) + (0.1 * hund)
    ajuste_der = (base * (1.0 + 0.2*bias)) + (0.1 * hund)

    # limpiar valores negativos
    ajuste_izq = ajuste_izq.clip(lower=0)
    ajuste_der = ajuste_der.clip(lower=0)

    df_state[LEFT_COL]  = ajuste_izq
    df_state[RIGHT_COL] = ajuste_der

    print("✔ Etiquetas automáticas generadas.")
else:
    print("✔ Se usarán etiquetas reales (no están vacías).")


# ===========================
# 3. Construir X, Y
# ===========================
feature_cols = [
    HUND_COL,
    ALIN_COL,
    ROLL_COL,
    PITCH_COL,
    YAW_COL,
    SPEED_COL,
]

# 1) Convertir etiquetas a numérico (forzando errores a NaN)
labels = df_state[[LEFT_COL, RIGHT_COL]].apply(
    pd.to_numeric,
    errors="coerce"   # cualquier cosa rara -> NaN
)

# 2) Crear máscara de filas válidas (sin NaN)
mask_valid = ~labels.isna().any(axis=1)

print(f"Filas totales: {len(df_state)}, filas válidas sin NaN en etiquetas: {mask_valid.sum()}")

# 3) Filtrar df_state y labels
df_state_clean = df_state[mask_valid].reset_index(drop=True)
labels_clean = labels[mask_valid].reset_index(drop=True)

# 4) Construir X, Y SOLO con filas válidas
X = df_state_clean[feature_cols].values.astype(np.float32)
Y = labels_clean.values.astype(np.float32)

# 5) Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

print("X_train:", X_train.shape, "Y_train:", Y_train.shape)
print("X_test :", X_test.shape,  "Y_test :", Y_test.shape)

# ===========================
# 4. Definir MLP
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo:", device)

model = MLPReg(in_dim=X_train.shape[1], hidden=[64,32], out_dim=2).to(device)
opt = optim.Adam(model.parameters(), lr=0.005)
loss_fn = nn.MSELoss()

Xtr_t = torch.from_numpy(X_train).to(device)
Ytr_t = torch.from_numpy(Y_train).to(device)
Xte_t = torch.from_numpy(X_test).to(device)
Yte_t = torch.from_numpy(Y_test).to(device)


# ===========================
# 5. Entrenamiento
# ===========================
epochs = 300
batch_size = 128
n = Xtr_t.shape[0]

model.train()
for ep in range(epochs):
    perm = np.random.permutation(n)
    epoch_loss = 0.0

    for i in range(0, n, batch_size):
        idx = perm[i:i+batch_size]
        xb = Xtr_t[idx]
        yb = Ytr_t[idx]

        opt.zero_grad()
        preds = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        opt.step()

        epoch_loss += loss.item() * xb.size(0)
    epoch_loss /= n

    if ep % 50 == 0 or ep == epochs-1:
        model.eval()
        with torch.no_grad():
            y_pred_test = model(Xte_t).cpu().numpy()
        mse = mean_squared_error(Y_test, y_pred_test)
        mae = mean_absolute_error(Y_test, y_pred_test)
        r2  = r2_score(Y_test, y_pred_test)
        print(f"Epoch {ep:03d} | loss_train={epoch_loss:.4f} "
              f"| test_MSE={mse:.4f} | test_MAE={mae:.4f} | test_R2={r2:.4f}")
        model.train()


# ===========================
# 6. Métricas + tiempo inferencia
# ===========================
model.eval()
with torch.no_grad():
    y_pred_test = model(Xte_t).cpu().numpy()

mse = mean_squared_error(Y_test, y_pred_test)
mae = mean_absolute_error(Y_test, y_pred_test)
r2  = r2_score(Y_test, y_pred_test)

runs = 200
start = time.time()
for _ in range(runs):
    with torch.no_grad():
        _ = model(Xte_t)
if device.type == "cuda":
    torch.cuda.synchronize()
end = time.time()

total_time = end - start
avg_batch = total_time / runs
avg_sample_ms = (avg_batch / Xte_t.shape[0]) * 1000.0

print("\n=== MÉTRICAS FINALES ===")
print(f"Test MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test R² : {r2:.4f}")
print(f"Tiempo medio inferencia: {avg_sample_ms:.6f} ms/muestra")


# ===========================
# 7. Salida por secciones (para la tamping)
# ===========================
L_SECCION = 3.0  # largo de la máquina en metros

dist  = df_state[M_COL].values.astype(float)
lat   = df_state[LAT_COL].values.astype(float)
lon   = df_state[LON_COL].values.astype(float)
hund  = df_state[HUND_COL].values.astype(float)
alin  = df_state[ALIN_COL].values.astype(float)
roll  = df_state[ROLL_COL].values.astype(float)
pitch = df_state[PITCH_COL].values.astype(float)
yaw   = df_state[YAW_COL].values.astype(float)
speed = df_state[SPEED_COL].values.astype(float)

start_track = dist.min()
end_track   = dist.max()

secciones_output = []
sec_id = 0

current_start = start_track
model.eval()

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

    x_sec = np.array([
        hund_prom,
        alin_prom,
        roll_mean,
        pitch_mean,
        yaw_mean,
        speed_mean
    ], dtype=np.float32)

    with torch.no_grad():
        x_t = torch.from_numpy(x_sec).unsqueeze(0).to(device)
        pred = model(x_t).cpu().numpy()[0]

    ajuste_izq = float(np.clip(pred[0], 0.0, 200.0))
    ajuste_der = float(np.clip(pred[1], 0.0, 200.0))

    secciones_output.append({
        "seccion_id": int(sec_id),
        "lat_inicio": round(lat_inicio, 8),
        "lat_fin": round(lat_fin, 8),
        "lon_inicio": round(lon_inicio, 8),
        "lon_fin": round(lon_fin, 8),
        "hundimiento_mm": round(hund_prom, 4),
        "ajuste_izquierdo_mm": round(ajuste_izq, 4),
        "ajuste_derecho_mm": round(ajuste_der, 4)
    })

    sec_id += 1
    current_start = current_end


OUTPUT_JSON = "../results/output_modelo2_sections.json"
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(secciones_output, f, indent=2, ensure_ascii=False)

print(f"\nSe generaron {len(secciones_output)} secciones.")
print(f"Guardado en: {OUTPUT_JSON}")
print("Primeras 3 secciones:")
for sec in secciones_output[:3]:
    print(sec)

# ===========================
# 8. Guardar modelo entrenado
# ===========================
save_path = "../models/modelo2_tamping.pt"
torch.save(model.state_dict(), save_path)
print(f"\nModelo guardado en: {save_path}")
