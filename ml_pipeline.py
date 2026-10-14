"""Pipeline entrenable que reproduce los 7 predictores del notebook SCF 2022.

La predicción solo se habilita con datos reales SCF y modelos entrenados.
Los modelos se vuelven a entrenar en una partición común; las métricas
pueden diferir ligeramente de las del notebook por evitar leakage de escalado.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SCF_URL = "https://www.federalreserve.gov/econres/files/scf2022s.zip"
COLUMNAS = [
    "Ingreso_Anual", "Tiene_Cuenta_Inversion", "Educacion",
    "Tiene_Acciones", "Tiene_Fondos", "Horizonte_Planeacion",
    "Conocimiento_Financiero",
]
PERFILES = ["Conservador", "Moderado", "Agresivo"]
COLUMNAS_ORIGEN = {
    "y1": "ID", "x3014": "Tolerancia_Riesgo", "x3008": "Horizonte_Planeacion",
    "x5931": "Educacion", "x5729": "Ingreso_Anual",
    "x7556": "Conocimiento_Financiero", "x3913": "Tiene_Acciones",
    "x3819": "Tiene_Fondos", "x3923": "Tiene_Cuenta_Inversion",
}


def descargar_scf(destino="data/scf2022s.zip") -> Path:
    """Descarga datos oficiales; puede usarse el ZIP descargado manualmente."""
    destino = Path(destino)
    if destino.exists() and zipfile.is_zipfile(destino):
        return destino
    destino.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(SCF_URL, timeout=90)
    response.raise_for_status()
    if not zipfile.is_zipfile(io.BytesIO(response.content)):
        raise ValueError("La descarga oficial no es un ZIP válido.")
    destino.write_bytes(response.content)
    return destino


def cargar_scf(ruta_zip: str | Path) -> pd.DataFrame:
    """Carga UNA sola imputación por hogar, como el notebook original."""
    with zipfile.ZipFile(ruta_zip) as z:
        archivos = [n for n in z.namelist() if n.lower().endswith("p22i6.dta")]
        if not archivos:
            raise ValueError("El ZIP no contiene p22i6.dta. Usa scf2022s.zip (SCF completo 2022).")
        with z.open(archivos[0]) as archivo:
            df = pd.read_stata(
                io.BytesIO(archivo.read()),
                columns=list(COLUMNAS_ORIGEN), convert_categoricals=False,
            )
    df = df.rename(columns=COLUMNAS_ORIGEN)
    df = df[df["ID"] % 10 == 1].copy()
    df["Perfil"] = df["Tolerancia_Riesgo"].map({1: 2, 2: 2, 3: 1, 4: 0})
    df = df.dropna(subset=["Perfil"]).copy()
    df["Perfil"] = df["Perfil"].astype(int)
    df["Ingreso_Anual"] = df["Ingreso_Anual"].replace({-1: 0, -9: np.nan})
    df["Conocimiento_Financiero"] = df["Conocimiento_Financiero"].replace(-1, 0)
    for name in ["Tiene_Acciones", "Tiene_Fondos", "Tiene_Cuenta_Inversion"]:
        df[name] = df[name].replace({1: 1, 5: 0})
    # No usamos X3014 como entrada: es la variable de la que salió la etiqueta.
    return df


def entrenar_modelos(ruta_zip: str | Path):
    df = cargar_scf(ruta_zip)
    X, y = df[COLUMNAS], df["Perfil"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    clasificaciones = {
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42),
        "XGBoost": XGBClassifier(eval_metric="mlogloss", random_state=42, n_jobs=2),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(5,), activation="relu", solver="adam", max_iter=1000, random_state=42),
        "Naive Bayes": GaussianNB(),
    }
    modelos, metricas = {}, {}
    for nombre, clasificador in clasificaciones.items():
        columnas = COLUMNAS[:2] if nombre == "Naive Bayes" else COLUMNAS
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("modelo", clasificador),
        ])
        pipe.fit(X_train[columnas], y_train)
        y_pred = pipe.predict(X_test[columnas])
        metricas[nombre] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "recall_clases": recall_score(y_test, y_pred, average=None, labels=[0, 1, 2], zero_division=0).tolist(),
            "matriz": confusion_matrix(y_test, y_pred, labels=[0, 1, 2]).tolist(),
        }
        modelos[nombre] = {"modelo": pipe, "columnas": columnas}
    return {
        "modelos": modelos, "metricas": metricas, "n_entrenamiento": len(X_train),
        "n_prueba": len(X_test), "n_total": len(df),
        "ingreso_p95": float(X_train["Ingreso_Anual"].dropna().quantile(.95)),
        "fuente": "SCF 2022 (Federal Reserve), una imputación; partición 70/30 estratificada",
    }


def inferir(bundle, modelo_nombre: str, respuestas: dict):
    registro = pd.DataFrame([respuestas], columns=COLUMNAS)
    predic = []
    for nombre, item in bundle["modelos"].items():
        pipe = item["modelo"]
        x = registro[item["columnas"]]
        codigo = int(pipe.predict(x)[0])
        probas = pipe.predict_proba(x)[0].tolist()
        predic.append({"nombre": nombre, "codigo": codigo, "perfil": PERFILES[codigo], "scores": probas})
    elegido = next(p for p in predic if p["nombre"] == modelo_nombre)
    return elegido, predic
