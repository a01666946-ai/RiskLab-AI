"""Preparación opcional de los modelos antes de abrir Streamlit.

Uso:
  python train.py
  python train.py --zip /ruta/scf2022s.zip
"""
from __future__ import annotations
import argparse
from pathlib import Path
from ml_pipeline import descargar_scf, entrenar_modelos

p=argparse.ArgumentParser()
p.add_argument('--zip', dest='ruta_zip', default=None, help='Ruta de scf2022s.zip descargado manualmente')
args=p.parse_args()
archivo=Path(args.ruta_zip) if args.ruta_zip else descargar_scf(Path(__file__).resolve().parent/'data/scf2022s.zip')
print(f'Leyendo: {archivo}')
out=entrenar_modelos(archivo)
print(f"N={out['n_total']} | train={out['n_entrenamiento']} | test={out['n_prueba']}")
for nombre,m in out['metricas'].items():
 print(f"{nombre:<16} accuracy={m['accuracy']:.2%}   F1 macro={m['f1_macro']:.2%}  recalls={[round(x,2) for x in m['recall_clases']]}")
print('\nLos modelos se entrenarán de nuevo y quedarán cacheados al abrir Streamlit.')
