import pandas as pd
import numpy as np

df = pd.read_csv("../../ai_error_corrector/datasets/datos_inspeccion_vias.csv")

df["Linea"] = np.select(
    [df.index <= 4999,
     (df.index >= 5000) & (df.index <= 9999),
     df.index >= 10000],
    ["L1", "L2", "L3"]
)

df.to_csv("../../../Frontend/src/resources/datos_inspeccion_vias.csv", index=False)
