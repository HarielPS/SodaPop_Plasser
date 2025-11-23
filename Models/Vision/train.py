import os
import torch
import pandas as pd
from PIL import Image
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
import torch.optim as optim
import random
import time


class SleepersDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, images_dir, transforms=None, zoom_prob=0.5):
        self.data = pd.read_csv(csv_file)
        self.images_dir = images_dir
        self.transforms = transforms
        self.zoom_prob = zoom_prob
        self.imgs = self.data["filename"].unique()[:300]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.images_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        records = self.data[self.data["filename"] == img_name]
        boxes = records[["xmin", "ymin", "xmax", "ymax"]].values.astype("float32")
        labels = torch.ones((len(records),), dtype=torch.int64)

        # ---------- ZOOM AUGMENTATION (20%) ----------
        if random.random() < self.zoom_prob:

            w, h = img.size

            # Tomar un crop del 60%–80% del tamaño original
            scale = random.uniform(0.4, 0.8)
            crop_w = int(w * scale)
            crop_h = int(h * scale)

            # Elegir esquina superior izquierda dentro del rango válido
            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)

            # Recortar imagen
            img = img.crop((x1, y1, x1 + crop_w, y1 + crop_h))

            # Ajustar boxes respecto al nuevo origen
            new_boxes = []
            for xmin, ymin, xmax, ymax in boxes:
                xmin = max(0, xmin - x1)
                ymin = max(0, ymin - y1)
                xmax = max(0, xmax - x1)
                ymax = max(0, ymax - y1)

                # Si el box cae fuera del crop se ignora
                if xmax > 0 and ymax > 0 and xmin < crop_w and ymin < crop_h:
                    new_boxes.append([xmin, ymin, xmax, ymax])

            # Si después del crop no quedó ninguna caja, usamos las originales
            if len(new_boxes) > 0:
                boxes = new_boxes

            # Reescalar a tamaño original
            img = img.resize((w, h))
            scale_x = w / crop_w
            scale_y = h / crop_h

            boxes = torch.tensor(
                [
                    [bx[0] * scale_x, bx[1] * scale_y, bx[2] * scale_x, bx[3] * scale_y]
                    for bx in boxes
                ],
                dtype=torch.float32,
            )
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # ---------- APPLY TRANSFORM ----------
        if self.transforms:
            img = self.transforms(img)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
        }

        return img, target


# Cargar Faster R-CNN preentrenado
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

# Cambiar la cabeza del detector (2 clases: fondo + sleeper)
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# GPU o CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)
model.to(device)

# Transform a aplicar en el dataset
transform = T.Compose(
    [
        T.ToTensor(),
    ]
)

# Dataset
dataset = SleepersDataset(
    "../../../../Datasets/Rieles/Dataset/train/_annotations.csv",
    "../../../../Datasets/Rieles/Dataset/train/",
    transforms=transform,
)

print("Dataset Cargado")

# DataLoader
data_loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=lambda x: tuple(zip(*x)),
)

optimizer = optim.SGD(
    model.parameters(),
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005,
)

num_epochs = 100
time_start = time.time()
for epoch in range(num_epochs):
    time_epoch_start = time.time()
    model.train()

    for imgs, targets in data_loader:

        # Enviar imágenes a GPU
        imgs = [img.to(device) for img in imgs]

        # Enviar targets a GPU
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward + pérdidas
        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            torch.save(
                model.state_dict(),
                f"../../../../Datasets/Rieles/Modelos/fasterrcnn_sleeper-epoch{epoch+1}.pth",
            )

    print(
        f"Epoch {epoch+1}, Loss: {losses.item():.4f}, Tiempo: {time.time() - time_epoch_start:.2f} segundos"
    )

end_time = time.time()
print(f"Tiempo total de entrenamiento: {end_time - time_start:.2f} segundos")
# Guardar modelo

torch.save(
    model.state_dict(), "../../../../Datasets/Rieles/Modelos/fasterrcnn_sleeper.pth"
)
