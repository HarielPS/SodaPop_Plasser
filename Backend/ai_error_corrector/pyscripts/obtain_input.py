import pandas as pd

df = pd.read_csv("../datasets/datos_inspeccion_vias.csv")

df.drop(
    columns=[
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
    ],
    inplace=True,
)

df.to_csv("../datasets/datos_input.csv", index=False)
