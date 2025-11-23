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
# NOTA: Asegúrate que input_modelo2_variables_estado.csv contiene las 4 nuevas columnas
FILE_STATE = "../datasets/input_modelo2_variables_estado.csv"  # generado por modelo 1
FILE_LABELS = "../datasets/plantilla_etiquetas_modelo2.csv"  # tú lo editas

df_state = pd.read_csv(FILE_STATE)
df_labels = pd.read_csv(FILE_LABELS)

print("Columnas estado:", df_state.columns.tolist())
print("Columnas etiquetas:", df_labels.columns.tolist())


# ===========================
# 2. Mapeo de columnas (¡Actualizado para rieles separados!)
# ===========================
TIME_COL = "Time"
LAT_COL = "X_pos_Lat"
LON_COL = "X_pos_Lon"
M_COL = "Y_pos_m"
ROLL_COL = "Roll_rad"
PITCH_COL = "Pitch_rad"
YAW_COL = "Yaw_rad"
SPEED_COL = "vx_ms"

# Variables de Desviación SEPARADAS
HUND_IZQ_COL = "Hundimiento_Izq_mm"
HUND_DER_COL = "Hundimiento_Der_mm"
ALIN_IZQ_COL = "Alineacion_Izq_mm"
ALIN_DER_COL = "Alineacion_Der_mm"

# Etiquetas de salida
LEFT_COL = "ajuste_izquierdo_mm"
RIGHT_COL = "ajuste_derecho_mm"


# ===========================
# 3. Validar columnas del dataset de estado
# ===========================
required_state_cols = [
    TIME_COL,
    LAT_COL,
    LON_COL,
    M_COL,
    HUND_IZQ_COL,
    HUND_DER_COL,
    ALIN_IZQ_COL,
    ALIN_DER_COL,  # 🛑 ACTUALIZADO
    ROLL_COL,
    PITCH_COL,
    YAW_COL,
    SPEED_COL,
]

for col in required_state_cols:
    if col not in df_state.columns:
        # Muestra un error más específico si las nuevas columnas no están
        if col in [HUND_IZQ_COL, HUND_DER_COL, ALIN_IZQ_COL, ALIN_DER_COL]:
            print("\n-------------------------------------------------------------")
            print(
                "⚠ ERROR: Las columnas de riel separado NO se encuentran en el archivo."
            )
            print("  Asegúrate de haber ejecutado el 'script_fisica_mejorado.py'")
            print("-------------------------------------------------------------")

        raise ValueError(f"Falta la columna '{col}' en {FILE_STATE}")

df_state = df_state.sort_values(M_COL).reset_index(drop=True)


# ===========================
# 4. Validar columnas de etiquetas y MERGE
# ===========================
required_label_cols = [M_COL, LEFT_COL, RIGHT_COL]

for col in required_label_cols:
    if col not in df_labels.columns:
        raise ValueError(f"Falta la columna '{col}' en {FILE_LABELS}")

# Nos quedamos SOLO con clave y etiquetas
df_labels_small = df_labels[[M_COL, LEFT_COL, RIGHT_COL]]

# MERGE por Y_pos_m (clave común)
df_state = df_state.merge(df_labels_small, on=M_COL, how="left")

print("\nDespués del merge:")
print(df_state.head())


# ===========================
# 5. Generar etiquetas automáticas si están vacías (¡Lógica actualizada!)
# ===========================
df_state[LEFT_COL] = pd.to_numeric(df_state[LEFT_COL], errors="coerce")
df_state[RIGHT_COL] = pd.to_numeric(df_state[RIGHT_COL], errors="coerce")

all_empty = df_state[[LEFT_COL, RIGHT_COL]].isna().all().all()

if all_empty:
    print("⚠ No hay etiquetas reales. Se generarán etiquetas automáticas...")

    # Usar los hundimientos separados (milimétricos)
    hund_izq = df_state[HUND_IZQ_COL].astype(float).clip(lower=0)
    hund_der = df_state[HUND_DER_COL].astype(float).clip(lower=0)
    alin_izq = df_state[ALIN_IZQ_COL].astype(float)
    alin_der = df_state[ALIN_DER_COL].astype(float)

    # Heurística de ajuste: 80% del hundimiento para corrección vertical,
    # más un 10% del valor absoluto de alineación para la corrección lateral
    ajuste_izq = hund_izq * 0.8 + alin_izq.abs() * 0.1
    ajuste_der = hund_der * 0.8 + alin_der.abs() * 0.1

    df_state[LEFT_COL] = ajuste_izq.clip(lower=0)
    df_state[RIGHT_COL] = ajuste_der.clip(lower=0)

    print("✔ Etiquetas automáticas generadas.")
else:
    print("✔ Se usarán etiquetas reales donde existan.")


# ===========================
# 6. Construir X, Y (¡Actualizado! 8 Features)
# ===========================
feature_cols = [
    HUND_IZQ_COL,  # Nuevo input
    HUND_DER_COL,  # Nuevo input
    ALIN_IZQ_COL,  # Nuevo input
    ALIN_DER_COL,  # Nuevo input
    ROLL_COL,
    PITCH_COL,
    YAW_COL,
    SPEED_COL,
]

labels = df_state[[LEFT_COL, RIGHT_COL]].apply(pd.to_numeric, errors="coerce")
mask_valid = ~labels.isna().any(axis=1)

print(
    f"Filas totales: {len(df_state)}, filas válidas con etiquetas: {mask_valid.sum()}"
)
print(
    f"El modelo MLP usará {len(feature_cols)} variables de entrada (Features)."
)  # 🛑 Comentario de Features

df_state_clean = df_state[mask_valid].reset_index(drop=True)
labels_clean = labels[mask_valid].reset_index(drop=True)

X = df_state_clean[feature_cols].values.astype(np.float32)
Y = labels_clean.values.astype(np.float32)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)


# ===========================
# 7. Definir MLP (In_dim se ajusta automáticamente a 8)
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

# in_dim = 8 (se calcula automáticamente)
model = MLPReg(in_dim=X_train.shape[1]).to(device)
opt = optim.Adam(model.parameters(), lr=0.005)
loss_fn = nn.MSELoss()

Xtr_t = torch.from_numpy(X_train).to(device)
Ytr_t = torch.from_numpy(Y_train).to(device)
Xte_t = torch.from_numpy(X_test).to(device)
Yte_t = torch.from_numpy(Y_test).to(device)


# ===========================
# 8. Entrenamiento (Sin cambios)
# ===========================
epochs = 300
batch_size = 128
n = Xtr_t.shape[0]

model.train()
for ep in range(epochs):
    perm = np.random.permutation(n)
    epoch_loss = 0.0

    for i in range(0, n, batch_size):
        idx = perm[i : i + batch_size]
        xb = Xtr_t[idx]
        yb = Ytr_t[idx]

        opt.zero_grad()
        preds = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        opt.step()

        epoch_loss += loss.item() * xb.size(0)
    epoch_loss /= n

    if ep % 50 == 0 or ep == epochs - 1:
        model.eval()
        with torch.no_grad():
            y_pred_test = model(Xte_t).cpu().numpy()
        mse = mean_squared_error(Y_test, y_pred_test)
        mae = mean_absolute_error(Y_test, y_pred_test)
        r2 = r2_score(Y_test, y_pred_test)
        print(
            f"Epoch {ep:03d} | loss={epoch_loss:.4f} | MSE={mse:.4f} | MAE={mae:.4f} | R2={r2:.4f}"
        )
        model.train()


# ===========================
# 9. Métricas finales (Sin cambios)
# ===========================
model.eval()
with torch.no_grad():
    y_pred_test = model(Xte_t).cpu().numpy()

mse = mean_squared_error(Y_test, y_pred_test)
mae = mean_absolute_error(Y_test, y_pred_test)
r2 = r2_score(Y_test, y_pred_test)

print("\n=== MÉTRICAS FINALES ===")
print("MSE:", mse)
print("MAE:", mae)
print("R2 :", r2)


# ===========================
# 10. Salida por secciones (¡Lógica de Features Actualizada y ION Reducida!)
# ===========================
# 🛑 CORRECCIÓN: Reducimos L_SECCION para generar más segmentos incluso en tramos cortos.
L_SECCION = 0.1

dist = df_state[M_COL].values
lat = df_state[LAT_COL].values
lon = df_state[LON_COL].values
hund_izq = df_state[HUND_IZQ_COL].values
hund_der = df_state[HUND_DER_COL].values
alin_izq = df_state[ALIN_IZQ_COL].values
alin_der = df_state[ALIN_DER_COL].values
roll = df_state[ROLL_COL].values
pitch = df_state[PITCH_COL].values
yaw = df_state[YAW_COL].values
speed = df_state[SPEED_COL].values

start_track = dist.min()
end_track = dist.max()

# 🛑 Diagnóstico de la longitud de la vía
total_length = end_track - start_track
print(f"\nLongitud total de la vía procesada: {total_length:.2f} metros.")
if total_length <= 3.0:
    print(
        "⚠ ¡Advertencia! El dataset de estado es muy corto (menos de 3m). Asegúrate de que el input del Modelo 2 contenga más datos de distancia (columna Y_pos_m)."
    )


secciones_output = []
sec_id = 0

current_start = start_track
model.eval()

while current_start < end_track:
    current_end = min(current_start + L_SECCION, end_track)
    mask = (dist >= current_start) & (dist <= current_end)

    # Esta lógica asegura que salte si no hay puntos en esa sección, aunque con L_SECCION=0.1 es raro.
    if not np.any(mask):
        current_start = current_end
        continue

    # Calcular promedios de las 8 features de entrada
    hund_izq_prom = float(max(0, hund_izq[mask].mean()))
    hund_der_prom = float(max(0, hund_der[mask].mean()))
    alin_izq_prom = float(alin_izq[mask].mean())
    alin_der_prom = float(alin_der[mask].mean())
    roll_mean = float(roll[mask].mean())
    pitch_mean = float(pitch[mask].mean())
    yaw_mean = float(yaw[mask].mean())
    speed_mean = float(speed[mask].mean())

    # 🛑 Creación del array de 8 features para el modelo
    x_sec = np.array(
        [
            hund_izq_prom,
            hund_der_prom,
            alin_izq_prom,
            alin_der_prom,
            roll_mean,
            pitch_mean,
            yaw_mean,
            speed_mean,
        ],
        dtype=np.float32,
    )

    with torch.no_grad():
        pred = model(torch.from_numpy(x_sec).unsqueeze(0).to(device)).cpu().numpy()[0]

    secciones_output.append(
        {
            "seccion_id": sec_id,
            "lat_inicio": float(lat[mask][0]),
            "lat_fin": float(lat[mask][-1]),
            "lon_inicio": float(lon[mask][0]),
            "lon_fin": float(lon[mask][-1]),
            "hundimiento_izquierdo_mm": hund_izq_prom,
            "hundimiento_derecho_mm": hund_der_prom,
            "ajuste_izquierdo_mm": float(np.clip(pred[0], 0, 200)),
            "ajuste_derecho_mm": float(np.clip(pred[1], 0, 200)),
        }
    )

    sec_id += 1
    current_start = current_end


OUTPUT_JSON = "../results/output_modelo2_sections.json"
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
json.dump(secciones_output, open(OUTPUT_JSON, "w"), indent=2)

print(f"\nSe generaron {len(secciones_output)} secciones.")
print("Guardado en:", OUTPUT_JSON)


# ===========================
# 11. Guardar modelo (Sin cambios)
# ===========================
save_path = "../models/modelo2_tamping.pt"
torch.save(model.state_dict(), save_path)
print("Modelo guardado en:", save_path)
