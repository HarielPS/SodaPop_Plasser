import pandas as pd

df = pd.read_csv("../datasets/datos_inspeccion_vias.csv")

df = df[["Lat_caro", "Lon_caro"]]

df["route_id"] = df.index // 5000


df.to_csv("../datasets/routes.csv", index=False)
