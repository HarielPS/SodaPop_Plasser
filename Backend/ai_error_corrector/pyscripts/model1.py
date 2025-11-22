import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib

# --- 1. Configuración de Hardware y Optimización ---
# Usamos float16 (AMP) para optimizar el rendimiento en la Jetson Nano (GPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Usando GPU: {device} (Activando float16/AMP)")
    use_amp = True
else:
    device = torch.device("cpu")
    print("Usando CPU (float32). float16/AMP inactivo.")
    use_amp = False

# --- 2. Carga y Preparación de Datos ---
FILE_PATH = "../datasets/datos_inspeccion_vias.csv"
data = pd.read_csv(FILE_PATH)

# Definición de las 14 variables de entrada y salida
INPUT_COLS = [col for col in data.columns if "barato" in col]
OUTPUT_COLS = [col for col in data.columns if "caro" in col]
X = data[INPUT_COLS].values
Y = data[OUTPUT_COLS].values

# Normalización (CRÍTICA para la convergencia del modelo)
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
scaler_Y = StandardScaler()
Y_scaled = scaler_Y.fit_transform(Y)

# División en conjuntos de entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y_scaled, test_size=0.2, random_state=42, shuffle=False
)

# Conversión a tensores y ajuste de dimensión (Batch, Secuencia, Caract.)
# Secuencia de longitud 1 para este dataset simple
X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)
Y_train_t = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1).to(device)
X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
Y_test_t = torch.tensor(Y_test, dtype=torch.float32).unsqueeze(1).to(device)


# --- 3. Definiciones de DataLoader y Modelo GRU ---
class SensorDataset(Dataset):
    """
    Clase para manejar el dataset de tensores.
    """

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


train_dataset = SensorDataset(X_train_t, Y_train_t)
test_dataset = SensorDataset(X_test_t, Y_test_t)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)


class FusionGRU(nn.Module):
    """
    Modelo GRU para la fusión y limpieza de sensores.
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(FusionGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU procesa la secuencia de datos (IMU, Láser, GNSS)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

        # Capa lineal para mapear la salida de la GRU a las 14 variables deseadas
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Inicializar el estado oculto (h0)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # out: (batch_size, seq_len, hidden_size)
        out, _ = self.gru(x, h0)

        # Tomar la salida del último paso de la secuencia y pasar por la capa lineal
        out = self.fc(out[:, -1, :])
        return out


INPUT_SIZE = X_train.shape[1]
OUTPUT_SIZE = Y_train.shape[1]
HIDDEN_SIZE = 64  # Valor bajo, óptimo para Jetson Nano
NUM_LAYERS = 2
model = FusionGRU(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS).to(device)

# --- 4. Entrenamiento ---
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 100
batch_size = 4

# Inicializar el escalador para AMP si usamos GPU
if use_amp:
    scaler = torch.cuda.amp.GradScaler()

print(f"Comenzando entrenamiento en {device} por {num_epochs} épocas...")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()

        if use_amp:
            # Entrenamiento con float16 (Automatic Mixed Precision)
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                # Squeeze(1) elimina la dimensión de secuencia (longitud 1)
                loss = criterion(outputs.squeeze(1), targets.squeeze(1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Entrenamiento estándar (float32)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(1), targets.squeeze(1))
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.6f}"
        )


# --- 5. Evaluación (MSE por Variable) ---
def evaluate_model(model, loader, scaler_Y, output_cols):
    """Calcula el MSE por variable de salida en el conjunto de prueba."""
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            all_predictions.append(outputs.cpu().numpy().squeeze())
            all_targets.append(targets.cpu().numpy().squeeze())

    # Concatenar todos los batches
    pred_scaled = np.concatenate(all_predictions, axis=0)
    target_scaled = np.concatenate(all_targets, axis=0)

    # Desnormalizar (CRUCIAL para obtener errores en unidades reales)
    pred_real = scaler_Y.inverse_transform(pred_scaled)
    target_real = scaler_Y.inverse_transform(target_scaled)

    # Calcular MSE por cada columna
    mse_results = {}
    for i, col in enumerate(output_cols):
        # MSE: (Predicción - Real)^2
        mse = mean_squared_error(target_real[:, i], pred_real[:, i])
        mse_results[col] = mse

    return mse_results


mse_per_column = evaluate_model(model, test_loader, scaler_Y, OUTPUT_COLS)
print("\n--- MSE por Variable (Precisión del Modelo 1) ---")
print("Un MSE bajo indica que la IA limpió bien el ruido del sensor.")
for col, mse in mse_per_column.items():
    # Usamos E-notation para mantener la precisión
    print(f"MSE para {col}: {mse:.6e}")

# --- 6. Guardar Modelo y Escaladores ---
# Mover el modelo a CPU antes de guardar, si se entrenó en GPU.
if use_amp:
    model.to(torch.float32).cpu()
else:
    model.cpu()

torch.save(model.state_dict(), "../models/fusion_gru_weights.pth")
joblib.dump(scaler_X, "../models/scaler_X.pkl")
joblib.dump(scaler_Y, "../models/scaler_Y.pkl")

print("\nModelo guardado como 'fusion_gru_weights.pth'")
print(
    "Escaladores guardados ('scaler_X.pkl' y 'scaler_Y.pkl') listos para la Jetson Nano."
)
