import os
import time
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim


# ===========================
# 1. Rutas
# ===========================
# CSV NUEVO con etiquetas reales (o refinadas)
# CSV NUEVO que quieres probar y etiquetar
NEW_STATE_CSV = "../datasets/estado_nueva_medicion_etiquetada.csv"

# Modelo base que ya entrenaste antes
BASE_MODEL_PATH = "../models/modelo2_tamping.pt"

# Dónde guardar el modelo actualizado
UPDATED_MODEL_PATH = "../models/modelo2_tamping_actualizado.pt"

# Opcional: resultados
OUTPUT_CSV  = "../results/estado_nueva_medicion_con_pred_update.csv"
OUTPUT_JSON = "../results/estado_nueva_medicion_sections_update.json"

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)


# ===========================
# 2. Definir arquitectura (igual que antes)
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
# 3. Cargar modelo base
# ===========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo para actualización:", device)

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

print(f"Cargando modelo base desde: {BASE_MODEL_PATH}")
state_dict = torch.load(BASE_MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
print("Modelo base cargado.")


# ===========================
# 4. Cargar nuevo CSV etiquetado
# ===========================
df_new = pd.read_csv(NEW_STATE_CSV)
print("Columnas del CSV nuevo:", df_new.columns.tolist())

# Columnas necesarias
NEEDED_COLS = [
    "Hundimiento_mm",
    "Alineacion_mm",
    "Roll_rad",
    "Pitch_rad",
    "Yaw_rad",
    "vx_ms",
    "ajuste_izquierdo_mm",
    "ajuste_derecho_mm",
]

for col in NEEDED_COLS:
    if col not in df_new.columns:
        raise ValueError(f"Falta la columna '{col}' en {NEW_STATE_CSV}")

# Convertir etiquetas a numérico y filtrar filas válidas
labels = df_new[["ajuste_izquierdo_mm", "ajuste_derecho_mm"]].apply(
    pd.to_numeric, errors="coerce"
)
mask_valid = ~labels.isna().any(axis=1)

print(f"Filas totales nuevo CSV: {len(df_new)}, filas válidas con etiquetas: {mask_valid.sum()}")

df_clean = df_new[mask_valid].reset_index(drop=True)
labels_clean = labels[mask_valid].reset_index(drop=True)

X = df_clean[feature_cols].astype(np.float32).values
Y = labels_clean.astype(np.float32).values

if len(df_clean) == 0:
    raise ValueError("No hay filas válidas con etiquetas numéricas en el nuevo CSV.")

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

print("X_train:", X_train.shape, "Y_train:", Y_train.shape)
print("X_test :", X_test.shape,  "Y_test :", Y_test.shape)


# ===========================
# 5. Fine-tuning (actualización)
# ===========================
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)  # lr más pequeño para fine-tuning
loss_fn = nn.MSELoss()

Xtr_t = torch.from_numpy(X_train).to(device)
Ytr_t = torch.from_numpy(Y_train).to(device)
Xte_t = torch.from_numpy(X_test).to(device)
Yte_t = torch.from_numpy(Y_test).to(device)

epochs = 150
batch_size = 64
n = Xtr_t.shape[0]

print("\nComenzando fine-tuning sobre el modelo base...\n")
model.train()
for ep in range(epochs):
    perm = np.random.permutation(n)
    epoch_loss = 0.0

    for i in range(0, n, batch_size):
        idx = perm[i:i+batch_size]
        xb = Xtr_t[idx]
        yb = Ytr_t[idx]

        optimizer.zero_grad()
        preds = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * xb.size(0)
    epoch_loss /= n

    if ep % 30 == 0 or ep == epochs - 1:
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
# 6. Métricas finales
# ===========================
model.eval()
with torch.no_grad():
    y_pred_test = model(Xte_t).cpu().numpy()

mse = mean_squared_error(Y_test, y_pred_test)
mae = mean_absolute_error(Y_test, y_pred_test)
r2  = r2_score(Y_test, y_pred_test)

print("\n=== MÉTRICAS FINALES (modelo actualizado) ===")
print(f"Test MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test R² : {r2:.4f}")


# ===========================
# 7. Guardar modelo actualizado
# ===========================
torch.save(model.state_dict(), UPDATED_MODEL_PATH)
print(f"\nModelo ACTUALIZADO guardado en: {UPDATED_MODEL_PATH}")


# ===========================
# 8. (Opcional) Inferencia sobre TODO el CSV nuevo y guardar resultados
# ===========================
X_all = df_new[feature_cols].astype(np.float32).values
X_all_t = torch.from_numpy(X_all).to(device)

with torch.no_grad():
    preds_all = model(X_all_t).cpu().numpy()

ajuste_izq_pred = np.clip(preds_all[:, 0], 0.0, 200.0)
ajuste_der_pred = np.clip(preds_all[:, 1], 0.0, 200.0)

df_new["ajuste_izquierdo_pred_mm"] = ajuste_izq_pred
df_new["ajuste_derecho_pred_mm"]   = ajuste_der_pred

df_new.to_csv(OUTPUT_CSV, index=False)
print(f"CSV con predicciones del modelo actualizado guardado en: {OUTPUT_CSV}")


# ===========================
# 9. (Opcional) Generar secciones de 40 m para la tamping
# ===========================
if all(col in df_new.columns for col in ["X_pos_Lat", "X_pos_Lon", "Y_pos_m"]):
    print("\nGenerando secciones de 40 m con el modelo actualizado...")

    L_SECCION = 40.0

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

    print(f"JSON de secciones (modelo actualizado) guardado en: {OUTPUT_JSON}")
    print("Primeras secciones:")
    for sec in secciones_output[:3]:
        print(sec)
else:
    print("\nNo se encontraron columnas X_pos_Lat / X_pos_Lon / Y_pos_m en el CSV nuevo; se omite generación de secciones.")
