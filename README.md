# RISKLAB AI — Tu portafolio ideal

Proyecto académico que une **clasificación de perfil de riesgo con Machine Learning** y **matching automático con carteras** construidas a partir del informe de portafolios proporcionado.

## El producto

1. El usuario responde **7 preguntas** (las 7 características seleccionadas en el notebook por información mutua).
2. Un clasificador **entrenado realmente sobre SCF 2022** predice Conservador / Moderado / Agresivo. Por defecto se usa SVM; en el lateral se puede seleccionar uno de los seis clasificadores.
3. La pantalla cambia automáticamente a **TU PERFIL** y **TU PORTAFOLIO IDEAL**. El capital indicado calcula montos en MXN, sin entrar al clasificador.
4. La cartera disponible muestra **pesos estratégicos, máximo Sharpe o mínima varianza**, rendimiento esperado Black-Litterman, volatilidad, Sharpe, activos y tendencias del informe, riesgo aportado, backtest y escenarios Monte Carlo precalculados.
5. El **Laboratorio de modelos** compara las métricas fuera de muestra de los seis algoritmos, muestra matrices de confusión y las siete variables.

### Importante: se clasifican perfiles, no rendimientos

El modelo se entrena para reproducir las etiquetas que el equipo derivó de la **tolerancia al riesgo X3014** en SCF 2022. Ese atributo nunca entra entre las siete variables predictoras. Accuracy/F1/recall evalúan la clasificación histórica; **no son probabilidades de rentabilidad ni de suitability para un cliente**. `predict_proba` no está calibrado adicionalmente, y las salidas por clase son orientativas.

### Carteras disponibles

- **Conservador** (A): 14 instrumentos, tres variantes. Rendimiento esperado estratégico 7.91% y volatilidad 7.31% según el informe.
- **Moderado** (B): 15 instrumentos, tres variantes. Rendimiento esperado estratégico 9.59% y volatilidad 11.82% según el informe.
- **Agresivo**: no aparece en `fuentes/resumen.md`; la app permite clasificarlo, pero **no fabrica una cartera agresiva**.

## Instalación en VS Code (Mac, Windows o Linux)

Abre la carpeta del proyecto, y en la terminal ejecuta:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

En Windows PowerShell, para activar el entorno usa `.venv\Scripts\Activate.ps1` en lugar de `source .venv/bin/activate`.

La app se abrirá normalmente en `http://localhost:8501`.

### Primera ejecución: entrenar los 6 clasificadores

La primera vez que pidas un resultado, el programa descarga automáticamente **scf2022s.zip** del sitio oficial de la Reserva Federal (alrededor de 9 MB), selecciona una imputación y entrena seis clasificadores en una partición de train/test del 70/30. Las sesiones sucesivas conservan los modelos en la memoria caché del servidor Streamlit.

Si la descarga se bloquea, descarga manualmente:

**https://www.federalreserve.gov/econres/files/scf2022s.zip**

Luego, súbela en el control que aparece al intentar clasificar; también puedes guardarla en `data/scf2022s.zip`. **No uses el CSV de extracto resumido:** no tiene las siete variables originales necesarias.

Para validar desde consola las métricas, ejecuta `python train.py` después de descargar el ZIP (o `python train.py --zip /ruta/al/scf2022s.zip`). Esta operación no conserva los modelos en disco: la app los entrena y cachea en memoria.

## En qué se parece y en qué difiere del notebook

Se respeta la misma selección de 7 predictores, la codificación de perfiles, `random_state=42`, `stratify=y`, las especificaciones principales de los seis algoritmos y la especificación de Naive Bayes con las dos variables usadas en su entrenamiento original. **Se corrige la fuga de información del escalado**: `StandardScaler` se ajusta solo sobre train mediante `Pipeline`, en lugar de utilizar estadísticas de todo el dataset antes de la división. Por ello, los resultados de la nueva evaluación pueden diferir ligeramente de las matrices del notebook.

**Monedas**: X5729 es ingreso **familiar total anual en USD de 2021**. Si se introduce ingreso en MXN, la app lo convierte usando un **tipo de cambio supuesto editable**; no se consulta el FX actual ni se hace un ajuste de inflación. El monto a invertir y resultados de cartera se expresan en MXN. SCF representa hogares estadounidenses y el clasificador **no está validado para México**.

**Portafolios**: valores tomados exclusivamente de `fuentes/resumen.md` con corte del 7/oct/2026; las señales técnicas llegan al 6/oct/2026 y no son en tiempo real. Para renta fija México el informe utiliza proxies y tasas mensuales FRED que pueden subestimar volatilidades y correlaciones. Las proyecciones compuestas a tasa constante son ilustrativas, no backtests ni Monte Carlo recalculado para el cliente.

## Archivos

```
app.py                    → Pantallas, Streamlit, diseño negro + neón azul
ml_pipeline.py            → SCF, limpieza, 6 modelos, inferencia
portafolios.json          → Pesos, métricas, backtests y técnicos extraídos del estudio
train.py                  → Prueba de entrenamiento en terminal
requirements.txt          → Dependencias
.streamlit/config.toml    → Tema
fuentes/resumen.md        → Documento original del portafolio
notebooks/*.ipynb         → Notebook original de los clasificadores
```

## GitHub y publicación

Este proyecto **no** se publica en GitHub Pages: Pages solo aloja contenido estático y **Streamlit ejecuta Python en un servidor**. Súbelo a GitHub y despliega con [Streamlit Community Cloud](https://share.streamlit.io/), apuntando a `app.py`. Es necesario que el servidor pueda descargar la base oficial durante el arranque o contar con un método de suministro seguro.

La app no guarda respuestas personales en archivos. Las respuestas viven en la sesión Streamlit y la descarga de la base es un conjunto público.

**Uso exclusivo académico / demostrativo. No constituye asesoría de inversión, recomendación personalizada regulada ni promesa de rentabilidad.**

Fuentes públicas: [Reserva Federal SCF 2022](https://www.federalreserve.gov/econres/scf_2022.htm), [Codebook 2022](https://www.federalreserve.gov/econres/files/codebk2022.txt).


## Actualización: portafolio Agresivo

Se añadió la cartera final robusta de 25 activos del notebook `RODRIGO_PORTAF_REFINITIV (1).ipynb`, celda de robustez sin OPPJ/VRT. Las métricas walk-forward de 2022–2025 son CAGR 26.56%, volatilidad 15.69%, Sharpe 1.33 y caída máxima -16.78%. El 9% de la proyección es un supuesto ilustrativo del notebook (escenarios 6/9/12%), NO un rendimiento esperado estimado de la cartera. Las tendencias de los activos agresivos no estaban disponibles. Los montos de los resultados aparecen con separadores de miles; las cajas de entrada de Streamlit siguen usando el formato nativo sin separadores durante la edición. No se convierte MXN a USD para la compra real de acciones estadounidenses.
