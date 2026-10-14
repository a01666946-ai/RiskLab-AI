"""Pruebas sin instalar Streamlit y sin descargar el SCF (datos sintéticos)."""
import io
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from ml_pipeline import COLUMNAS, cargar_scf, entrenar_modelos, inferir


def prueba_datos():
    data=json.loads(Path(__file__).with_name('portafolios.json').read_text(encoding='utf-8'))
    for perfil,cartera in data['perfiles'].items():
        for nombre,pesos in cartera['allocations'].items():
            assert abs(sum(pesos.values())-100)<.02,(perfil,nombre,sum(pesos.values()))
            assert all(p>=0 for p in pesos.values())
    assert 'Agresivo' not in data['perfiles']
    print('✓ Carteras: pesos verificados; perfil agresivo no inventado')


def prueba_inferencia():
    r=np.random.default_rng(10)
    n=240
    dataframe=pd.DataFrame({
        'y1':np.arange(n)*10+1, 'x3014':r.choice([1,2,3,4],n),
        'x3008':r.integers(1,6,n),'x5931':r.integers(8,15,n),
        'x5729':r.integers(10000,250000,n),'x7556':r.integers(1,11,n),
        'x3913':r.choice([1,5],n),'x3819':r.choice([1,5],n),
        'x3923':r.choice([1,5],n),
    })
    with tempfile.TemporaryDirectory() as p:
        ruta=Path(p)
        dataframe.to_stata(ruta/'p22i6.dta',write_index=False)
        with zipfile.ZipFile(ruta/'scf2022s.zip','w') as z:
            z.write(ruta/'p22i6.dta','p22i6.dta')
        bundle=entrenar_modelos(ruta/'scf2022s.zip')
        entrada=dict(zip(COLUMNAS,[50000,1,12,0,1,4,5]))
        pred,votos=inferir(bundle,'SVM',entrada)
        assert pred['perfil'] in ['Conservador','Moderado','Agresivo']
        assert len(votos)==6
        assert abs(sum(pred['scores'])-1)<1e-6
    print('✓ SCF STATA, preprocesamiento, seis modelos e inferencia probados con datos SINTÉTICOS (solo test)')


if __name__=='__main__':
    prueba_datos()
    prueba_inferencia()
