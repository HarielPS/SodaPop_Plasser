import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

# ==============================================================================
# --- CONFIGURACIÓN GENERAL Y RUTAS DE ARCHIVOS ---
# ==============================================================================

INFERENCE_FILE = "../dataset/datos_input.csv"
OUTPUT_JSON = "../results/output_modelo2_sections.json"

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
print(f"Dispositivo: {device}")


# ==============================================================================
# --- 1. MODELO GRU (FUSIÓN/LIMPIEZA DE SENSORES) ---
# ==============================================================================


class FusionGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(FusionGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # h0 shape: (num_layers, batch, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.gru(x, h0)
        return self.fc(out[:, -1, :])


def run_modelo1(df_crudo: pd.DataFrame) -> pd.DataFrame:
    print("\n--- INICIO: MODELO 1 (FUSIÓN/LIMPIEZA) ---")

    INPUT_SIZE = 14
    OUTPUT_SIZE = 14
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2

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

    try:
        model_gru = FusionGRU(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS).to(
            device
        )
        model_gru.load_state_dict(
            torch.load("../models/fusion_gru_weights.pth", map_location=device)
        )
        model_gru.eval()
        scaler_X = joblib.load("../models/scaler_X.pkl")
        scaler_Y = joblib.load("../models/scaler_Y.pkl")
        print("✔ Modelo GRU cargado y escaladores también.")
    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}")
        return None

    time_data = df_crudo[TIME_COL]
    X = df_crudo.drop(TIME_COL, axis=1).values
    X_scaled = scaler_X.transform(X)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1).to(device)
    with torch.no_grad():
        outputs_scaled = model_gru(X_tensor)

    outputs_scaled = outputs_scaled.squeeze(1).cpu().numpy()
    outputs_real = scaler_Y.inverse_transform(outputs_scaled)

    df_limpio = pd.DataFrame(outputs_real, columns=output_cols_names)
    df_limpio.insert(0, TIME_COL, time_data)

    print(f"✔ Modelo 1 completado. Filas: {df_limpio.shape[0]}")
    return df_limpio


# ==============================================================================
# --- 2. MODELO FÍSICO ---
# ==============================================================================


def run_fisica(df_limpio: pd.DataFrame) -> pd.DataFrame:
    print("\n--- PASO 2: CÁLCULO DE FÍSICA Y ESTADO ---")

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
    df_estado["vy_ms"] = df_clean["vU"]
    df_estado["vz_ms"] = df_clean["vN"]

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

    print("✔ Variables de estado generadas.")
    return df_final


# ==============================================================================
# --- 3. MODELO 2 (MLP POR SECCIONES) ---
# ==============================================================================


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
        return self.net(x)


def run_modelo2(df_state: pd.DataFrame):
    print("\n--- INICIO: MODELO 2 (PREDICCIÓN DE AJUSTE) ---")

    try:
        model = MLPReg(in_dim=len(FEATURE_COLS_MODELO2)).to(device)
        model.load_state_dict(
            torch.load("../models/modelo2_tamping.pt", map_location=device)
        )
        model.eval()
        print("✔ Modelo 2 cargado.")
    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}")
        return

    dist = df_state[M_COL].values
    lat = df_state[LAT_COL].values
    lon = df_state[LON_COL].values

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

        with torch.no_grad():
            pred = (
                model(torch.from_numpy(x_sec).unsqueeze(0).to(device)).cpu().numpy()[0]
            )

        secciones_output.append(
            {
                "seccion_id": sec_id,
                "lat_inicio": float(lat[mask][0]),
                "lat_fin": float(lat[mask][-1]),
                "lon_inicio": float(lon[mask][0]),
                "lon_fin": float(lon[mask][-1]),
                "hundimiento_izquierdo_mm": float(hund_izq_prom),
                "hundimiento_derecho_mm": float(hund_der_prom),
                "ajuste_izquierdo_mm": float(np.clip(pred[0], 0, 200)),
                "ajuste_derecho_mm": float(np.clip(pred[1], 0, 200)),
            }
        )

        sec_id += 1
        current_start = current_end

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    json.dump(secciones_output, open(OUTPUT_JSON, "w"), indent=2)

    print(f"✔ Modelo 2 completado. Secciones generadas: {len(secciones_output)}")
    print(f"✔ JSON guardado en {OUTPUT_JSON}")


# ==============================================================================
# --- PIPELINE PRINCIPAL ---
# ==============================================================================
if __name__ == "__main__":
    print(f"Iniciando Pipeline de Inferencia desde: {INFERENCE_FILE}")

    try:
        df_crudo = pd.read_csv(INFERENCE_FILE)
        print(f"✔ Datos de entrada cargados. Filas: {df_crudo.shape[0]}")
    except FileNotFoundError:
        print(f"❌ ERROR: No se encontró {INFERENCE_FILE}")
        exit()

    df_limpio = run_modelo1(df_crudo)
    if df_limpio is None:
        exit()

    df_estado = run_fisica(df_limpio)
    if df_estado is None:
        exit()

    run_modelo2(df_estado)

    print("\nFIN DEL PROCESO.")
    print("\nFIN DEL PROCESO.")
