import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import torchvision.transforms as T
import cv2
import numpy as np

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Modelo preentrenado base
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

# Reemplazar la cabeza de clasificación
num_classes = 2  # 0=background, 1=sleeper
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Cargar pesos entrenados
model.load_state_dict(
    torch.load(
        "../../../../Datasets/Rieles/Modelos/fasterrcnn_sleeper.pth",
        map_location=device,
    )
)
model.to(device)
model.eval()

# Cargar imagen con PIL
img_path = "../../../../Datasets/Rieles/Dataset/test/prueba3.jpg"
img = Image.open(img_path).convert("RGB")

# Transformar a tensor
transform = T.Compose(
    [
        T.ToTensor(),
        # T.Resize((360, 640)),
    ]  # tamaño que usaste en entrenamiento
)
img_tensor = transform(img).to(device)

with torch.no_grad():
    predictions = model([img_tensor])


# Convertir PIL → NumPy para OpenCV
img_cv = np.array(img)
img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
# img_cv = cv2.resize(img_cv, (640, 360))


boxes = predictions[0]["boxes"]
labels = predictions[0]["labels"]
scores = predictions[0]["scores"]

for box, label, score in zip(boxes, labels, scores):
    if score > 0.5:
        x1, y1, x2, y2 = box.int().tolist()  # <-- convertir a lista de ints
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img_cv,
            f"Sleeper {score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

# Mostrar
cv2.imshow("Predicción", img_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()
