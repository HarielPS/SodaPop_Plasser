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
device = torch.device("cpu")

# ===========================================
#  VARIABLES PARA CARGA DIFERIDA
# ===========================================
faster_model = None
transform = T.Compose([T.ToTensor()])


# ===========================================
#  CARGA DIFERIDA DEL MODELO FASTER R-CNN
# ===========================================
def init_faster_rcnn():
    """
    Carga el modelo Faster R-CNN SOLO cuando se llama esta función.
    Django NO lo ejecuta en el import.
    """

    global faster_model

    if faster_model is not None:
        return faster_model

    print("Inicializando modelo Faster R-CNN...")

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    MODEL_PATH = BASE_DIR / "modelos" / "fasterrcnn_sleeper.pth"

    if not MODEL_PATH.exists():
        print("⚠ ADVERTENCIA: Modelo Faster R-CNN no encontrado.")
        print("⚠ Saltando carga de modelo (modo safe).")
        faster_model = None
        return None

    model.load_state_dict(
        torch.load(str(MODEL_PATH), map_location=device)
    )

    model.to(device)
    model.eval()

    faster_model = model
    print("✔ Modelo Faster R-CNN cargado exitosamente.")
    return faster_model


# ===========================================
#  FUNCIÓN DE DETECCIÓN (solo si se init el modelo)
# ===========================================

def detectar_dormideros_cv(image_cv):
    """
    Detecta dormideros usando Faster R-CNN.
    Si no se cargó el modelo → devuelve la imagen original.
    """

    model = init_faster_rcnn()

    if model is None:
        print("⚠ Modelo no cargado. Retornando imagen sin procesar.")
        return image_cv

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
# ⚠ MODELOS 1 Y 2 (TAMPING)
# SE MOVIERON A CARGA DIFERIDA TAMBIÉN
# ===========================================

_model_gru = None
_scaler_X = None
_scaler_Y = None
_model_mlp = None


def init_tamping_models():
    """
    Inicializa GRU + MLP + scalers SOLO cuando se llama.
    No al iniciar Django.
    """
    global _model_gru, _scaler_X, _scaler_Y, _model_mlp

    if _model_gru is not None:
        return True

    try:
        print("Inicializando modelos GRU/MLP...")

        MODELS_DIR = BASE_DIR / "modelos"

        _model_gru = FusionGRU(14, 128, 14, 2).to(device)
        _model_gru.load_state_dict(
            torch.load(str(MODELS_DIR / "fusion_gru_weights.pth"), map_location=device)
        )
        _model_gru.eval()

        _scaler_X = joblib.load(MODELS_DIR / "scaler_X.pkl")
        _scaler_Y = joblib.load(MODELS_DIR / "scaler_Y.pkl")

        _model_mlp = MLPReg(in_dim=8).to(device)
        _model_mlp.load_state_dict(
            torch.load(str(MODELS_DIR / "modelo2_tamping.pt"), map_location=device)
        )
        _model_mlp.eval()

        print("✔ Modelos Tamping cargados.")
        return True

    except Exception as e:
        print(f"⚠ No se pudieron cargar los modelos Tamping: {e}")
        return False


# ===========================================
#  TODAS LAS FUNCIONES DE TAMPING NO SE EJECUTAN EN STARTUP
# ===========================================

def get_tamping_predictions(df_raw):
    """
    Se ejecuta SOLO cuando tú llames esta función.
    """
    if not init_tamping_models():
        return {"error": "Modelos tamper no disponibles."}

    # ... (tu lógica original aquí)
    return {"ok": True}


# ===========================================
#  FUNCIÓN FINAL PARA TESTS
# ===========================================

def final_tamping_predictions(INFERENCE_FILE):
    try:
        df_raw = pd.read_csv(INFERENCE_FILE)
        return get_tamping_predictions(df_raw)
    except Exception as e:
        return {"error": str(e)}
