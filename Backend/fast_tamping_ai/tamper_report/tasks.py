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
from django.conf import settings

BASE_DIR = Path(__file__).resolve().parent.parent

# ===========================================
#  CARGA DEL MODELO UNA SOLA VEZ
# ===========================================
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

print("Cargando modelo Faster R-CNN...")
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
MODELO_FASTER_RCNN_PATH = BASE_DIR / "modelos" / "fasterrcnn_sleeper.pth"

model.load_state_dict(
    torch.load(
        MODELO_FASTER_RCNN_PATH
        map_location=device,
    )
)

model.to(device)
model.eval()
print("✔ Modelo cargado correctamente")


# Transformación
transform = T.Compose([T.ToTensor()])


# ===========================================
#  FUNCIÓN QUE RECIBIRÁ LA IMAGEN
# ===========================================
def detectar_dormideros_cv(image_cv):
    """
    image_cv: imagen OpenCV (BGR)
    Return: imagen OpenCV (BGR) con bounding boxes dibujados
    """

    # Convert OpenCV → PIL
    img_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # Transformar
    img_tensor = transform(pil_img).to(device)

    # Inferencia
    with torch.no_grad():
        preds = model([img_tensor])

    boxes = preds[0]["boxes"].detach().cpu().numpy()
    scores = preds[0]["scores"].detach().cpu().numpy()

    # Dibujar cajas sobre copia
    result_img = image_cv.copy()

    for box, score in zip(boxes, scores):
        if score >= 0.5:
            x1, y1, x2, y2 = box.astype(int).tolist()

            # Dibujar bounding box sin texto
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Regresamos la imagen con cajas dibujadas
    return result_img


import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from typing import List, Dict, Union

# ==============================================================================
# --- CONFIGURACIÓN Y RUTAS DE ARCHIVOS ---
# ==============================================================================

# Se asume que los modelos y escaladores están en estas rutas relativas a donde se ejecute la API.
MODELS_DIR = "C:/Users/jlflo/Documents/Hack_Austria/SodaPop_Plasser/Models/Tamping_detector/models"

# Columnas base
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

# Variables del Modelo 2
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

L_SECCION = 0.1  # Longitud 10 cm por sección

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Dispositivo: {device}") # Se omite print en el módulo API

# ==============================================================================
# --- DEFINICIONES DE MODELOS (Necesarias para cargar) ---
# ==============================================================================


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


# ==============================================================================
# --- CACHE DE RECURSOS (Para evitar recargar modelos en cada llamada) ---
# ==============================================================================
# Variables globales para almacenar modelos y escaladores cargados
_model_gru = None
_scaler_X = None
_scaler_Y = None
_model_mlp = None


def _load_resources():
    """Carga los modelos y escaladores una sola vez."""
    global _model_gru, _scaler_X, _scaler_Y, _model_mlp

    if _model_gru is not None and _model_mlp is not None:
        return True  # Ya cargados

    try:
        # --- Modelo 1 (GRU) ---
        INPUT_SIZE = 14
        OUTPUT_SIZE = 14
        HIDDEN_SIZE = 128
        NUM_LAYERS = 2
        _model_gru = FusionGRU(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS).to(
            device
        )
        _model_gru.load_state_dict(
            torch.load(
                os.path.join(MODELS_DIR, "fusion_gru_weights.pth"), map_location=device
            )
        )
        _model_gru.eval()
        _scaler_X = joblib.load(os.path.join(MODELS_DIR, "scaler_X.pkl"))
        _scaler_Y = joblib.load(os.path.join(MODELS_DIR, "scaler_Y.pkl"))

        # --- Modelo 2 (MLP) ---
        _model_mlp = MLPReg(in_dim=len(FEATURE_COLS_MODELO2)).to(device)
        _model_mlp.load_state_dict(
            torch.load(
                os.path.join(MODELS_DIR, "modelo2_tamping.pt"), map_location=device
            )
        )
        _model_mlp.eval()

        print("✔ Modelos y escaladores cargados.")
        return True
    except FileNotFoundError as e:
        print(f"❌ ERROR al cargar recursos: {e}")
        return False
    except Exception as e:
        print(f"❌ ERROR inesperado al cargar recursos: {e}")
        return False


# ==============================================================================
# --- FUNCIONES DE INFERENCIA (Extraídas y refactorizadas) ---
# ==============================================================================


def _run_modelo1(df_crudo: pd.DataFrame) -> Union[pd.DataFrame, None]:
    """Ejecuta el Modelo 1 (Fusión/Limpieza) con los modelos cacheados."""
    model_gru = _model_gru
    scaler_X = _scaler_X
    scaler_Y = _scaler_Y

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

    time_data = df_crudo[TIME_COL]
    # Asegurarse de que no falta ninguna columna, aunque el script original asume que todas están presentes.
    try:
        X = df_crudo.drop(TIME_COL, axis=1).values
    except KeyError:
        # Esto manejaría el caso en que falta la columna de tiempo, pero se asume que existe
        return None

    X_scaled = scaler_X.transform(X)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1).to(device)
    with torch.no_grad():
        outputs_scaled = model_gru(X_tensor)

    outputs_scaled = outputs_scaled.squeeze(1).cpu().numpy()
    outputs_real = scaler_Y.inverse_transform(outputs_scaled)

    df_limpio = pd.DataFrame(outputs_real, columns=output_cols_names)
    df_limpio.insert(0, TIME_COL, time_data.reset_index(drop=True))

    return df_limpio


def _run_fisica(df_limpio: pd.DataFrame) -> pd.DataFrame:
    """Calcula las variables de física y estado."""
    W = 1435
    Y_IDEAL_M = 2240.35

    df_clean = df_limpio.copy()
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

    df_estado = pd.DataFrame()
    df_estado[TIME_COL] = df_clean["Time"]
    df_estado[M_COL] = df_clean["Alt"]
    df_estado["Z_pos_m"] = np.cumsum(df_clean["vN"] * df_clean["Time"].diff().fillna(0))
    df_estado["Z_pos_m"] -= df_estado["Z_pos_m"].iloc[0]

    df_estado[LAT_COL] = df_clean["Lat"]
    df_estado[LON_COL] = df_clean["Lon"]

    df_estado[SPEED_COL] = df_clean["vE"]
    df_estado[ROLL_COL] = np.arctan((df_clean["d_der"] - df_clean["d_izq"]) / W)
    df_estado[PITCH_COL] = df_clean["wy"]
    df_estado[YAW_COL] = df_clean["wz"]

    df_estado[HUND_IZQ_COL] = (Y_IDEAL_M - df_clean["Alt"]) * 1000 - (
        df_clean["d_izq"] - 500
    )
    df_estado[HUND_DER_COL] = (Y_IDEAL_M - df_clean["Alt"]) * 1000 - (
        df_clean["d_der"] - 500
    )

    df_estado[ALIN_IZQ_COL] = df_clean["ax"] * 500
    df_estado[ALIN_DER_COL] = -df_clean["ax"] * 500

    df_final = (
        df_estado[
            [
                TIME_COL,
                LAT_COL,
                LON_COL,
                M_COL,
                SPEED_COL,
                ROLL_COL,
                PITCH_COL,
                YAW_COL,
                HUND_IZQ_COL,
                HUND_DER_COL,
                ALIN_IZQ_COL,
                ALIN_DER_COL,
            ]
        ]
        .sort_values(M_COL)
        .reset_index(drop=True)
    )

    return df_final


def _run_modelo2(df_state: pd.DataFrame) -> List[Dict]:
    """Ejecuta el Modelo 2 (MLP) por secciones y devuelve el resultado JSON."""
    model = _model_mlp

    dist = df_state[M_COL].values
    lat = df_state[LAT_COL].values
    lon = df_state[LON_COL].values

    # Extraer las características una vez
    feature_data = {col: df_state[col].values for col in FEATURE_COLS_MODELO2}

    start_track = dist.min()
    end_track = dist.max()

    secciones_output = []
    sec_id = 0
    current_start = start_track

    while current_start < end_track:
        current_end = min(current_start + L_SECCION, end_track)
        mask = (dist >= current_start) & (dist <= current_end)

        if not np.any(mask):
            current_start = current_end
            continue

        # Calcular promedios de características para la sección
        x_sec = np.array(
            [
                float(max(0, feature_data[HUND_IZQ_COL][mask].mean())),
                float(max(0, feature_data[HUND_DER_COL][mask].mean())),
                float(feature_data[ALIN_IZQ_COL][mask].mean()),
                float(feature_data[ALIN_DER_COL][mask].mean()),
                float(feature_data[ROLL_COL][mask].mean()),
                float(feature_data[PITCH_COL][mask].mean()),
                float(feature_data[YAW_COL][mask].mean()),
                float(feature_data[SPEED_COL][mask].mean()),
            ],
            dtype=np.float32,
        )

        hund_izq_prom = x_sec[0]
        hund_der_prom = x_sec[1]

        # Predicción del modelo
        with torch.no_grad():
            pred = (
                model(torch.from_numpy(x_sec).unsqueeze(0).to(device)).cpu().numpy()[0]
            )

        # Generar diccionario de sección
        secciones_output.append(
            {
                "start_latitude": float(lat[mask][0]),
                "stop_latitude": float(lat[mask][-1]),
                "start_longitude": float(lon[mask][0]),
                "stop_longitude": float(lon[mask][-1]),
                "lift_left_mm": float(hund_izq_prom),
                "lift_right_mm": float(hund_der_prom),
                "adjustement_left_mm": float(np.clip(pred[0], 0, 200)),
                "adjustement_right_mm": float(np.clip(pred[1], 0, 200)),
            }
        )

        sec_id += 1
        current_start = current_end

    return secciones_output


# ==============================================================================
# --- FUNCIÓN PRINCIPAL DE LA API ---
# ==============================================================================


def get_tamping_predictions(df_crudo: pd.DataFrame) -> Union[List[Dict], Dict]:
    """
    Ejecuta el pipeline de modelos (GRU, Física, MLP) y devuelve las
    predicciones de ajuste de apisonamiento por sección.

    Args:
        df_crudo (pd.DataFrame): DataFrame de entrada con datos crudos.
                                 Debe contener la columna 'Time' y las 14
                                 columnas requeridas para el Modelo 1.

    Returns:
        Union[List[Dict], Dict]: Una lista de diccionarios (JSON) con los
                                 resultados por sección, o un diccionario de
                                 error en caso de fallo.
    """
    # 1. Cargar recursos (solo la primera vez)
    if not _load_resources():
        return {
            "error": "No se pudieron cargar los modelos o escaladores. Verifique las rutas en la carpeta 'models'."
        }

    # 2. Modelo 1: Fusión/Limpieza de Sensores
    df_limpio = _run_modelo1(df_crudo)
    if df_limpio is None:
        return {
            "error": "Error durante la ejecución del Modelo 1. Verifique las columnas de entrada."
        }

    # 3. Cálculo de Física y Estado
    df_estado = _run_fisica(df_limpio)
    # df_estado nunca debería ser None si df_limpio no lo es.

    # 4. Modelo 2: Predicción de Ajuste (Tamping)
    resultados_json = _run_modelo2(df_estado)

    # El script original guardaba el JSON, aquí simplemente lo devolvemos
    return resultados_json


def final_tamping_predictions(INFERENCE_FILE):
    # Ejemplo de uso (solo para pruebas)
    # Se recomienda usar un framework web como Flask o FastAPI para una API real.

    # Simulación de carga de datos (reemplazar con tu propia lógica de carga)

    try:
        # Esto solo funcionará si los archivos existen en la ruta relativa.
        # En un entorno de API real, los datos vendrían de una petición (POST/PUT).
        df_crudo_test = pd.read_csv(INFERENCE_FILE)
        print(f"Iniciando API con datos de prueba: {df_crudo_test.shape[0]} filas.")

        # Ejecutar la API
        predictions = get_tamping_predictions(df_crudo_test)

        # Imprimir o procesar la salida JSON
        if isinstance(predictions, list):
            return predictions  # Retorna el JSON para mandarlo con la API
        else:
            print(f"❌ Fallo en la API: {predictions.get('error')}")

    except FileNotFoundError:
        print(
            f"❌ ERROR: Archivo de prueba no encontrado en {INFERENCE_FILE}. No se pudo simular la API."
        )
    except Exception as e:
        print(f"❌ ERROR Inesperado en la prueba de la API: {e}")
