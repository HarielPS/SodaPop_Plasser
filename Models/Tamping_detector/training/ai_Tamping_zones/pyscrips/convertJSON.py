import json

# === 1. Cargar JSON de entrada desde archivo ===
with open("../results/output_modelo2_sections.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# === 2. Mapa genérico de conversión ===
key_map = {
    "lat_inicio": "start_latitude",
    "lon_inicio": "start_longitude",
    "lat_fin": "end_latitude",
    "lon_fin": "end_longitude",
    "hundimiento_izquierdo_mm": "lift_left_mm",
    "hundimiento_derecho_mm": "lift_right_mm",
    "ajuste_izquierdo_mm": "adjustment_left_mm",
    "ajuste_derecho_mm": "adjustment_right_mm",
}

# === 3. Conversión usando el mapa ===
converted = [{key_map[k]: v for k, v in item.items() if k in key_map} for item in data]

# === 4. Guardar resultado en output.json ===
with open("../results/output.json", "w", encoding="utf-8") as f:
    json.dump(converted, f, indent=2, ensure_ascii=False)

print("Archivo output.json generado con éxito")
