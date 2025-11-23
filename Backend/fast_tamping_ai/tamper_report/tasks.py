# app/tasks.py

from pathlib import Path
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import torchvision.transforms as T
import cv2
import numpy as np
import os
import json
import pandas as pd
import torch.nn as nn
import joblib
from typing import List, Dict, Union
from django.conf import settings


# ===========================================
#  CONFIGURACIÓN BASE
# ===========================================

BASE_DIR = Path(__file__).resolve().parent.parent

# Render NO tiene GPU → seteamos CPU fijo
device = torch.device("cpu")


# ===========================================
#  CARGA DEL MODELO FASTER R-CNN UNA SOLA VEZ
# ===========================================

print("Cargando modelo Faster R-CNN...")
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

MODELO_FASTER_RCNN_PATH = BASE_DIR / "modelos" / "fasterrcnn_sleeper.pth"

# FIX DE COMA AQUÍ ✔✔✔
model.load_state_dict(
    torch.load(
        str(MODELO_FASTER_RCNN_PATH),
        map_location=device,
    )
)

model.to(device)
model.eval()
print("✔ Modelo Faster R-CNN cargado correctamente")


# Transformación a Tensor
transform = T.Compose([T.ToTensor()])


# ===========================================
#  FUNCIÓN DE DETECCIÓN (MODELO 3)
# ===========================================

def detectar_dormideros_cv(image_cv):
    """
    Detecta dormideros usando Faster R-CNN.
    image_cv → imagen OpenCV (BGR)
    Devuelve imagen con bounding boxes.
    """
    img_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    img_tensor = transform(pil_img).to(device)

    with torch.no_grad():
        preds = model([img_tensor])

    boxes = preds[0]["boxes"].cpu().numpy()
    scores = preds[0]["scores"].cpu().numpy()

    result_img = image_cv.copy()

    for box, score in zip(boxes, scores):
        if score >= 0.5:
            x1, y1, x2, y2 = box.astype(int).tolist()
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return result_img


# ===========================================
#  CONFIGURACIÓN MODELOS 1 Y 2 (TAMPING)
# ===========================================

MODELS_DIR = BASE_DIR / "modelos"

TIME_COL = "Time"
M_COL = "Y_pos_m"
LAT_COL = "X_pos_Lat"
LON_COL = "X_pos_Lon"
ROLL_COL = "Roll_rad"
PITCH_COL = "Pitch_rad"
YAW_COL = "Yaw_rad"
SPEED_COL = "vx_ms"
HUND_IZQ_COL = "Hundimiento_Izq_mm"
HUND_DER_COL = "Hundimiento_Der_mm"
ALIN_IZQ_COL = "Alineacion_Izq_mm"
ALIN_DER_COL = "Alineacion_Der_mm"

FEATURE_COLS_MODELO2 = [
    HUND_IZQ_COL,
    HUND_DER_COL,
    ALIN_IZQ_COL,
    ALIN_DER_COL,
    ROLL_COL,
    PITCH_COL,
    YAW_COL,
    SPEED_COL,
]

L_SECCION = 0.1  # 10 cm


# ===========================================
#  DEFINICIONES DE MODELOS
# ===========================================

class FusionGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(FusionGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.gru(x, h0)
        return self.fc(out[:, -1, :])


class MLPReg(nn.Module):
    def __init__(self, in_dim, hidden=[64, 32], out_dim=2):
        super(MLPReg, self).__init__()
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


# ===========================================
#  CACHE DE MODELOS PARA NO RECARGAR
# ===========================================

_model_gru = None
_scaler_X = None
_scaler_Y = None
_model_mlp = None


def _load_resources():
    """Carga GRU + MLP + scalers una sola vez en memoria."""
    global _model_gru, _scaler_X, _scaler_Y, _model_mlp

    if _model_gru is not None:
        return True

    try:
        # Modelo GRU
        _model_gru = FusionGRU(14, 128, 14, 2).to(device)
        _model_gru.load_state_dict(
            torch.load(str(MODELS_DIR / "fusion_gru_weights.pth"), map_location=device)
        )
        _model_gru.eval()

        # Scalers
        _scaler_X = joblib.load(MODELS_DIR / "scaler_X.pkl")
        _scaler_Y = joblib.load(MODELS_DIR / "scaler_Y.pkl")

        # Modelo MLP (Modelo 2)
        _model_mlp = MLPReg(in_dim=len(FEATURE_COLS_MODELO2)).to(device)
        _model_mlp.load_state_dict(
            torch.load(str(MODELS_DIR / "modelo2_tamping.pt"), map_location=device)
        )
        _model_mlp.eval()

        print("✔ Modelos GRU / MLP cargados.")
        return True

    except Exception as e:
        print(f"❌ Error cargando modelos: {e}")
        return False


# ===========================================
#  MODELO 1: FUSIÓN / LIMPIEZA
# ===========================================

def _run_modelo1(df):
    """
    Ejecuta el Modelo 1 (GRU).
    """
    X = df.drop(TIME_COL, axis=1).values
    X_scaled = _scaler_X.transform(X)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        out_scaled = _model_gru(X_tensor)

    out = _scaler_Y.inverse_transform(out_scaled.cpu().numpy())

    cols = [
        "ax_caro", "ay_caro", "az_caro",
        "wx_caro", "wy_caro", "wz_caro",
        "Lat_caro", "Lon_caro", "Alt_caro",
        "vN_caro", "vE_caro", "vU_caro",
        "d_izq_caro", "d_der_caro"
    ]

    df2 = pd.DataFrame(out, columns=cols)
    df2.insert(0, TIME_COL, df[TIME_COL].values)

    return df2


# ===========================================
#  MODELO DE FÍSICA
# ===========================================

def _run_fisica(df):
    W = 1435
    Y_IDEAL = 2240.35

    df_out = pd.DataFrame()
    df_out[TIME_COL] = df["Time"]
    df_out[M_COL] = df["Alt_caro"]

    df_out[LAT_COL] = df["Lat_caro"]
    df_out[LON_COL] = df["Lon_caro"]

    df_out[SPEED_COL] = df["vE_caro"]
    df_out[ROLL_COL] = np.arctan((df["d_der_caro"] - df["d_izq_caro"]) / W)
    df_out[PITCH_COL] = df["wy_caro"]
    df_out[YAW_COL] = df["wz_caro"]

    df_out[HUND_IZQ_COL] = (Y_IDEAL - df["Alt_caro"]) * 1000 - (df["d_izq_caro"] - 500)
    df_out[HUND_DER_COL] = (Y_IDEAL - df["Alt_caro"]) * 1000 - (df["d_der_caro"] - 500)

    df_out[ALIN_IZQ_COL] = df["ax_caro"] * 500
    df_out[ALIN_DER_COL] = -df["ax_caro"] * 500

    return df_out.sort_values(M_COL).reset_index(drop=True)


# ===========================================
#  MODELO 2: PREDICCIÓN SECCIONES
# ===========================================

def _run_modelo2(df):
    dist = df[M_COL].values
    lat = df[LAT_COL].values
    lon = df[LON_COL].values

    features = {col: df[col].values for col in FEATURE_COLS_MODELO2}

    start = dist.min()
    end = dist.max()

    output = []

    sec = start
    while sec < end:
        sec_end = min(sec + L_SECCION, end)
        mask = (dist >= sec) & (dist <= sec_end)

        if np.any(mask):
            x = np.array([features[c][mask].mean() for c in FEATURE_COLS_MODELO2], dtype=np.float32)

            with torch.no_grad():
                pred = _model_mlp(torch.tensor(x).unsqueeze(0).to(device)).cpu().numpy()[0]

            output.append({
                "start_latitude": float(lat[mask][0]),
                "stop_latitude": float(lat[mask][-1]),
                "start_longitude": float(lon[mask][0]),
                "stop_longitude": float(lon[mask][-1]),
                "lift_left_mm": float(x[0]),
                "lift_right_mm": float(x[1]),
                "adjustement_left_mm": float(np.clip(pred[0], 0, 200)),
                "adjustement_right_mm": float(np.clip(pred[1], 0, 200)),
            })

        sec += L_SECCION

    return output


# ===========================================
#  FUNCIÓN PRINCIPAL
# ===========================================

def get_tamping_predictions(df_raw):
    if not _load_resources():
        return {"error": "Error cargando modelos"}

    df1 = _run_modelo1(df_raw)
    df2 = _run_fisica(df1)
    return _run_modelo2(df2)


# ===========================================
#  FUNCIÓN FINAL PARA TEST
# ===========================================

def final_tamping_predictions(INFERENCE_FILE):
    try:
        df_raw = pd.read_csv(INFERENCE_FILE)
        return get_tamping_predictions(df_raw)
    except Exception as e:
        return {"error": str(e)}
