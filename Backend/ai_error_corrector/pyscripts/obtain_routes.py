import pandas as pd

df = pd.read_csv("../datasets/datos_limpios_modelo1_output.csv")

df = df[["Lat_caro", "Lon_caro"]]


df.to_csv("../datasets/routes.csv", index=False)
