"""RISKLAB AI — Cuestionario → ML real → portafolio documentado."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ml_pipeline import COLUMNAS, PERFILES, descargar_scf, entrenar_modelos, inferir

BASE = Path(__file__).resolve().parent
DATOS = json.loads((BASE / "portafolios.json").read_text(encoding="utf-8"))
BLUE, CYAN, VIOLET, RED = "#278AFF", "#04E0FF", "#AD8BFF", "#F67B99"
BG, PAPER, MUTED = "#050915", "#0D1730", "#9FB3D2"

st.set_page_config(page_title="RISKLAB AI | Tu portafolio ideal", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');
html, body, [class*="css"], [data-testid="stApp"] {font-family:'DM Sans',sans-serif;}
[data-testid="stApp"] {background:radial-gradient(circle at 80% -10%,#0B2760 0,#071125 30%,#050915 63%); color:#EAF4FF;}
[data-testid="stSidebar"] {background:#091124;border-right:1px solid #1A3A67;}
[data-testid="stHeader"] {background:transparent;}
.block-container {max-width:1400px;padding-top:2.0rem;padding-bottom:4rem;}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif;color:#F5FBFF;letter-spacing:-.025em;}
p {color:#C9D9F1;} small, .stCaption {color:#96ABCD!important;}
[data-testid="stMetric"] {background:linear-gradient(145deg,#101D38,#0A1429);border:1px solid #25406B;border-radius:15px;padding:17px 19px;}
[data-testid="stMetricLabel"] {color:#9BB4D5;}
[data-testid="stMetricValue"] {color:#F1FAFF; font-family:'Space Grotesk';}
[data-testid="stVerticalBlockBorderWrapper"] {border-color:#213B63!important;border-radius:17px!important;background:linear-gradient(135deg,rgba(14,27,51,.80),rgba(8,16,36,.80));}
.stButton>button[kind="primary"], .stFormSubmitButton>button[kind="primary"] {background:linear-gradient(100deg,#087CE9,#06C8EF)!important;color:#001527!important;border:0!important;border-radius:10px!important;font-weight:800!important;box-shadow:0 0 22px #0ABEFF45;min-height:3.1rem;}
.stButton>button {border-radius:10px;}
[data-baseweb="select"]>div,[data-baseweb="input"]>div{background:#0D1A31!important;border-color:#31527D!important;}
[data-testid="stProgressBar"] div[role="progressbar"] {background:linear-gradient(90deg,#1878FF,#06DDFF)!important;}
.hero {padding:32px 34px;border-radius:22px;position:relative;overflow:hidden;border:1px solid #28588E;background:radial-gradient(circle at 90% 20%,#123B79 0,transparent 43%),linear-gradient(115deg,#0E2347,#071329);box-shadow:0 20px 60px #0005;}
.hero::before {content:'';position:absolute;width:230px;height:230px;border:1px solid #29D8FF50;border-radius:50%;right:-80px;top:-92px;box-shadow:0 0 0 48px #1F79E514,0 0 0 97px #1F79E50D;pointer-events:none;}
.hero h1 {font-size:clamp(34px,4.1vw,59px);line-height:1.07;margin:14px 0 15px;max-width:850px;}
.hero p {font-size:1.08rem;max-width:740px;color:#ADC6E7;}
.pill {display:inline-block;border:1px solid #2487B9;background:#073F5C7F;padding:5px 12px;border-radius:999px;letter-spacing:1.5px;font-size:11px;font-weight:800;color:#4AE7FF;}
.section-label {font-size:12px;letter-spacing:2px;font-weight:800;color:#06CFFF;margin-top:28px;margin-bottom:5px;}
.profile-box {padding:29px;border:1px solid #17ADFA;background:linear-gradient(110deg,#0B2853 10%,#081A35 73%);border-radius:20px;box-shadow:0 0 35px #046CC72A;}
.profile-box h1 {font-size:clamp(36px,4vw,57px);color:#4DE9FF;margin:9px 0;}
.profile-box p {color:#C7DCFF;max-width:820px;}
.profile-box .tag {color:#4DE9FF;font-size:12px;letter-spacing:2px;font-weight:800;}
.asset-title{font-weight:750;font-size:14px;color:#DCEBFF;}
.disclaimer{border-left:3px solid #3E9BFF;background:#0A2142;padding:13px 18px;margin:14px 0;border-radius:6px;color:#BBD2EC;font-size:13px;}
div[data-testid="stDataFrame"] {border:1px solid #1E416A;border-radius:12px;}
.stTabs [data-baseweb="tab"] {color:#A7C4E8;font-weight:750;}
hr {border-color:#1A3556;}
@media(max-width:700px){.hero{padding:24px 20px}.block-container{padding-left:1rem;padding-right:1rem;}}
</style>
""", unsafe_allow_html=True)


def html(t): st.markdown(t, unsafe_allow_html=True)
def moneda(x): return f"${x:,.0f} MXN"
def numero(x): return f"{x:,.0f}"


def leer_monto(texto):
    """Acepta cantidades como 10000, 10,000 o $10,000.50."""
    dato = str(texto).replace(",", "").replace("$", "").replace(" ", "").strip()
    valor = float(dato)
    if not np.isfinite(valor) or valor < 0:
        raise ValueError("Introduce una cantidad numérica válida y positiva.")
    return valor
def formato(x): return f"{x:.2f}%"


def titulo_seccion(kicker, titulo, bajada=None):
    html(f'<div class="section-label">{kicker.upper()}</div>')
    st.subheader(titulo)
    if bajada: st.caption(bajada)


def look(fig, height=330):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#CEE2FB", size=12),
        margin=dict(l=10,r=10,t=12,b=10), height=height,
        legend=dict(orientation="h", y=-0.13),
    )
    fig.update_xaxes(gridcolor="#1D3453", zerolinecolor="#304462")
    fig.update_yaxes(gridcolor="#1D3453", zerolinecolor="#304462")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})


@st.cache_resource(show_spinner=False)
def modelo_entrenado(ruta: str, tamano: int):
    return entrenar_modelos(ruta)


def obtener_modelos():
    ruta = descargar_scf(BASE / "data/scf2022s.zip")
    return modelo_entrenado(str(ruta), ruta.stat().st_size)


TIPOS = {
    "CETES":"Renta fija MX", "MBONO29":"Renta fija MX", "MBONO34":"Renta fija MX", "UDI35":"Renta fija MX",
    "SGOV":"Renta fija internacional", "IEI":"Renta fija internacional", "TIP":"Renta fija internacional",
    "IGSB":"Renta fija internacional", "EMB":"Renta fija internacional", "GLDM":"Oro",
}


def tipo_activo(ticker): return TIPOS.get(ticker,"Renta variable")


def tabla_asignacion(pesos, capital):
    r=[]
    for ticker, peso in pesos.items():
        if peso <= 0: continue
        t = DATOS["tecnico"].get(ticker,{})
        r.append({"Ticker":ticker,"Categoría":tipo_activo(ticker),"Peso (%)":round(peso,2),
                  "Monto (MXN)":round(capital*peso/100,2),"Tendencia":t.get("tendencia","Sin dato"),
                  "RSI (14)":t.get("rsi",np.nan),"Cruce":t.get("cruce","—")})
    return pd.DataFrame(r).sort_values("Peso (%)",ascending=False)


# Navegación persistente y elecciones de modelo.
if "fase" not in st.session_state: st.session_state.fase = "cuestionario"
with st.sidebar:
    html('<div style="font-family:Space Grotesk;font-size:25px;font-weight:800;color:#EAF7FF">◈ RISK<span style="color:#06D8FF">LAB</span><span style="font-size:12px;color:#6FA8DA">  AI</span></div>')
    st.caption("INTELLIGENT INVESTMENT MATCHING")
    st.divider()
    st.markdown("**Motor de clasificación**")
    nombre_modelo = st.selectbox("Modelo principal", ["SVM","XGBoost","KNN","Decision Tree","MLP","Naive Bayes"],
                                 help="Puedes comparar los seis algoritmos del notebook. SVM es la opción inicial por su accuracy de referencia.")
    st.caption("Las probabilidades de clase no son certeza de inversión.")
    st.divider()
    pagina = st.radio("Explorar",["Mi experiencia", "Laboratorio de modelos"], label_visibility="visible")
    st.divider()
    if st.button("↻ Volver al cuestionario", use_container_width=True):
        st.session_state.fase="cuestionario"
        st.rerun()
    st.caption("SCF 2022 · 4,595 hogares · 7 variables · 6 algoritmos")
    st.caption("Investigación académica · No constituye asesoría individual.")


def cuestionario():
    html('''<div class="hero"><div class="pill">EXPERIENCIA PERSONALIZADA • MACHINE LEARNING</div>
    <h1>Invierte según<br><span style="color:#36E6FF">tu perfil.</span> No al azar.</h1>
    <p>Responde siete preguntas financieras. Nuestros clasificadores analizarán tus respuestas y te conectarán con una cartera construida en el proyecto.</p>
    <div style="color:#91BCE5;font-size:12px;margin-top:20px">01 · CONÓCETE &nbsp;&nbsp;→&nbsp;&nbsp; 02 · INTELIGENCIA ARTIFICIAL &nbsp;&nbsp;→&nbsp;&nbsp; 03 · TU PORTAFOLIO</div></div>''')
    titulo_seccion("01 / 03 — Diagnóstico financiero", "Construyamos tu perfil", "Las preguntas corresponden a las 7 variables con mayor información mutua del notebook SCF 2022.")
    st.progress(0.33)
    with st.form("perfil_form", clear_on_submit=False):
        with st.container(border=True):
            st.markdown("#### 01 — Tu situación financiera")
            c1,c2=st.columns(2)
            with c1:
                ingresos_texto=st.text_input("Ingreso familiar anual (antes de impuestos)",value="360,000",
                                            help="Puedes escribir 360,000 o 360000. Equivalente del ingreso familiar total.")
                moneda_ingreso=st.selectbox("Moneda del ingreso",["MXN", "USD"])
                tc=st.number_input("Tipo de cambio MXN por USD (supuesto editable)",min_value=1.0,value=18.0,step=.25,
                                   help="El SCF entrenó los modelos con ingresos familiares de Estados Unidos expresados en USD de 2021. No es una cotización actual.") if moneda_ingreso=="MXN" else 1.0
            with c2:
                educaciones={"Primaria":"2", "Secundaria":"3", "Preparatoria / bachillerato":"8",
                            "Universidad sin concluir":"9", "Técnico superior":"10", "Licenciatura":"12",
                            "Maestría":"13", "Doctorado / posgrado profesional":"14"}
                educ=st.selectbox("Nivel máximo de estudios completado",list(educaciones),index=5,
                                  help="Se mapea a la codificación X5931 de la encuesta SCF 2022.")
                conocimiento=st.slider("¿Qué tanto conoces de finanzas personales?",0,10,5,
                                       help="Autopercepción: 0=ningún conocimiento; 10=muy alto (X7556).")
            st.divider()
            st.markdown("#### 02 — Tu experiencia con inversiones")
            a,b,c=st.columns(3)
            with a: cuenta=st.radio("¿Tienes una cuenta de inversión?",["Sí","No"],index=1)
            with b: acciones=st.radio("¿Actualmente tienes acciones?",["Sí","No"],index=1)
            with c: fondos=st.radio("¿Actualmente tienes fondos de inversión?",["Sí","No"],index=1)
            st.divider()
            st.markdown("#### 03 — Tu horizonte de planeación")
            horizontes={"Próximos meses":1,"Próximo año":2,"Próximos años (2–4)":3,
                        "Entre 5 y 10 años":4,"Más de 10 años":5}
            horizonte=st.select_slider("Al planear tus ahorros y gastos, ¿qué plazo tiene más importancia para ti?",
                                        options=list(horizontes),value="Entre 5 y 10 años",
                                        help="Es el horizonte de planeación financiera X3008, no necesariamente el plazo de esta inversión.")
            st.divider()
            st.markdown("#### 04 — Simulación de inversión")
            c1,c2=st.columns(2)
            with c1:
                capital_texto=st.text_input("¿Cuánto quieres invertir? (MXN)",value="100,000",
                                              help="Puedes escribir 10,000 o 100,000 con separadores de miles.")
            with c2: plazo=st.selectbox("Plazo de visualización",[1,3,5,10],index=2,
                                        help="Solo cambia la proyección ilustrativa, no el perfil de riesgo.")
            st.caption("El capital y el plazo de visualización NO son variables del clasificador. Nunca preguntamos directamente «¿eres conservador o agresivo?»: eso es lo que predice el modelo.")
            enviar=st.form_submit_button("✦ DESCUBRIR MI PORTAFOLIO IDEAL",use_container_width=True,type="primary")
        if enviar:
            try:
                ingresos = leer_monto(ingresos_texto)
                capital = leer_monto(capital_texto)
                if capital < 1000:
                    st.error("El capital mínimo de esta simulación es $1,000 MXN.")
                    st.stop()
            except ValueError:
                st.error("Revisa los montos. Utiliza números como 10,000 o 125,500.50.")
                st.stop()
            st.session_state.respuestas={
                "Ingreso_Anual":float(ingresos/tc),
                "Tiene_Cuenta_Inversion":int(cuenta=="Sí"),
                "Educacion":int(educaciones[educ]),
                "Tiene_Acciones":int(acciones=="Sí"),
                "Tiene_Fondos":int(fondos=="Sí"),
                "Horizonte_Planeacion":horizontes[horizonte],
                "Conocimiento_Financiero":int(conocimiento),
            }
            st.session_state.capital=int(capital)
            st.session_state.plazo=int(plazo)
            st.session_state.fase="resultado"
            st.rerun()


def pantalla_error(exc):
    st.error("Para clasificar con Machine Learning REAL, primero hay que entrenar los modelos con el SCF 2022.")
    st.caption("La descarga automática no se completó. Puedes descargar el ZIP oficial y cargarlo aquí. No se sustituirá por un clasificador inventado.")
    st.code("https://www.federalreserve.gov/econres/files/scf2022s.zip", language="text")
    with st.expander("Ver detalle técnico"):
        st.write(str(exc))
    archivo=st.file_uploader("Sube scf2022s.zip (archivo oficial completo, no el extracto CSV)",type=["zip"])
    if archivo and st.button("Entrenar clasificadores con mi ZIP",type="primary"):
        import zipfile,io
        if not zipfile.is_zipfile(io.BytesIO(archivo.getvalue())):
            st.error("El archivo no es un ZIP válido.")
        else:
            carpeta=BASE/"data"
            carpeta.mkdir(exist_ok=True)
            (carpeta/"scf2022s.zip").write_bytes(archivo.getvalue())
            modelo_entrenado.clear()
            st.rerun()


def mostrar_resultado(bundle):
    elegido, predicciones=inferir(bundle,nombre_modelo,st.session_state.respuestas)
    perfil=elegido["perfil"]
    metros=bundle["metricas"][nombre_modelo]
    recall=metros["recall_clases"][elegido["codigo"]]
    acuerdo=sum(p["codigo"]==elegido["codigo"] for p in predicciones)
    html('''<div class="hero"><div class="pill">MATCH GENERADO CON ML REAL</div>
    <h1>Diseñado para <span style="color:#32E7FF">tu perfil.</span></h1>
    <p>Tu clasificación proviene de las siete respuestas y del algoritmo seleccionado, no de reglas fijas.</p></div>''')
    st.write("")
    html(f'''<div class="profile-box"><div class="tag">02 / 03 · TU DIAGNÓSTICO DE RIESGO</div>
    <h1>{perfil.upper()}</h1><p>El modelo <b>{nombre_modelo}</b> identificó este perfil con los datos proporcionados. Puedes comprobar abajo el respaldo estadístico y cómo votaron los demás clasificadores.</p></div>''')
    st.write("")
    k1,k2,k3,k4=st.columns(4)
    k1.metric("Accuracy de prueba",formato(100*metros["accuracy"]))
    k2.metric("Recall de este perfil",formato(100*recall))
    k3.metric("F1 macro del modelo",formato(100*metros["f1_macro"]))
    k4.metric("Acuerdo entre modelos",f"{acuerdo} de 6")
    html('<div class="disclaimer"><b>Interpretación correcta:</b> las métricas anteriores describen la evaluación histórica del clasificador. No son la probabilidad de que un portafolio gane dinero ni garantizan que este cliente esté bien clasificado.</div>')

    with st.expander("🧠 ¿Qué dijeron los 6 modelos? — Ver explicación", expanded=False):
        c1,c2=st.columns([1.2,1])
        with c1:
            valores=pd.DataFrame({"Perfil":PERFILES,"Puntuación no calibrada":np.array(elegido["scores"])*100})
            fig=px.bar(valores,x="Puntuación no calibrada",y="Perfil",orientation="h",color="Perfil",
                       color_discrete_map={"Conservador":CYAN,"Moderado":BLUE,"Agresivo":VIOLET},text="Puntuación no calibrada")
            fig.update_traces(texttemplate="%{text:.1f}%",textposition="outside")
            fig.update_layout(showlegend=False,xaxis_range=[0,110])
            look(fig,260)
            st.caption("Salidas predict_proba orientativas sin calibración adicional. La predicción final utiliza predict() del modelo.")
        with c2:
            votos=pd.DataFrame([{"Modelo":p["nombre"],"Clasificación":p["perfil"]} for p in predicciones])
            st.dataframe(votos,hide_index=True,use_container_width=True)
            st.caption("El acuerdo entre clasificadores no equivale a una probabilidad estadística de acierto.")

    st.markdown("---")
    titulo_seccion("03 / 03 — Tu match financiero", "TU PORTAFOLIO IDEAL", "Composición y escenarios extraídos del estudio de portafolios, con cálculos de montos personalizados.")

    cartera=DATOS["perfiles"][perfil]
    if perfil == "Agresivo":
        opciones={"Final robusta (Refinitiv, 25 activos)":"Final_Validado"}
    else:
        opciones={"Estratégica (estudio original)":"Estrategico", "Máximo Sharpe (Black-Litterman)":"Max_Sharpe",
                  "Mínima varianza":"Min_Varianza"}
    estrategia=st.selectbox("Elige la estrategia de asignación",list(opciones),index=0,
                            help="Se usan exactamente los pesos del estudio; no se ejecuta una optimización nueva.")
    clave=opciones[estrategia]
    pesos=cartera["allocations"][clave]
    metricas=cartera["metricas"][clave]
    capital=st.session_state.capital
    anos=st.session_state.plazo
    suma=sum(pesos.values())
    if abs(suma-100)>.02: st.error("Los pesos del portafolio no suman 100%. Revisa los datos.");return
    m1,m2,m3,m4=st.columns(4)
    if perfil == "Agresivo":
        st.info("Portafolio agresivo: 25 activos del notebook Refinitiv. El 9% usado para proyectar es un escenario ilustrativo, no una expectativa Black-Litterman. Volatilidad y Sharpe provienen del walk-forward 2022–2025.")
    m1.metric("Escenario anual ilustrativo" if perfil=="Agresivo" else "Rendimiento anual esperado (BL)",formato(metricas["rendimiento"]))
    m2.metric("Volatilidad histórica WF" if perfil=="Agresivo" else "Volatilidad estimada",formato(metricas["volatilidad"]))
    m3.metric("Sharpe histórico WF" if perfil=="Agresivo" else "Sharpe esperado",f'{metricas["sharpe"]:.2f}')
    m4.metric(f"Valor ilustrativo a {anos} años",moneda(capital*(1+metricas["rendimiento"]/100)**anos))
    st.caption("Proyección de tasa constante, no garantizada. Para el perfil agresivo, la tasa base 9% es un supuesto ilustrativo del notebook y las métricas de riesgo corresponden al walk-forward; las otras carteras usan estimaciones ex ante Black-Litterman.")

    df=tabla_asignacion(pesos,capital)
    t1,t2,t3=st.tabs(["◉ Distribución de cartera", "↗ Proyección y escenarios", "◎ Tendencia y activos"])
    with t1:
        c1,c2=st.columns([.9,1.1])
        with c1:
            st.markdown("#### Asignación por clase de activo")
            grupos=df.groupby("Categoría",as_index=False)["Peso (%)"].sum()
            fig=px.pie(grupos,names="Categoría",values="Peso (%)",hole=.67,
                       color="Categoría",color_discrete_map={"Renta fija MX":"#16C0E8","Renta fija internacional":"#3B7DF5","Renta variable":"#A38AFE","Oro":"#F5C35C"})
            fig.update_traces(textinfo="percent",hovertemplate="%{label}<br>%{value:.1f}%<extra></extra>")
            fig.update_layout(annotations=[dict(text="ASIGNACIÓN",showarrow=False,font=dict(size=16,color="#C6E8FF"))])
            look(fig,365)
        with c2:
            st.markdown("#### Peso de cada instrumento")
            fig=px.bar(df.sort_values("Peso (%)"),x="Peso (%)",y="Ticker",orientation="h",color="Categoría",
                       color_discrete_map={"Renta fija MX":"#16C0E8","Renta fija internacional":"#3B7DF5","Renta variable":"#A38AFE","Oro":"#F5C35C"})
            fig.update_layout(legend=dict(orientation="h",y=-.30))
            look(fig,440)
        st.markdown("#### ¿Cuánto invertirías en cada activo?")
        tabla_visual=df[["Ticker","Categoría","Peso (%)","Monto (MXN)"]].copy()
        tabla_visual["Peso (%)"]=tabla_visual["Peso (%)"].map(lambda n: f"{n:,.2f}%")
        tabla_visual["Monto (MXN)"]=tabla_visual["Monto (MXN)"].map(moneda)
        st.dataframe(tabla_visual,use_container_width=True,hide_index=True)
        st.download_button("↓ Descargar asignación en CSV",df.to_csv(index=False).encode('utf-8-sig'),
                           file_name=f"RiskLab_{perfil}_{clave}.csv",mime="text/csv")
    with t2:
        a,b=st.columns([1.25,.75])
        with a:
            st.markdown("#### Evolución teórica de tu capital")
            x=np.arange(anos+1)
            y=capital*(1+metricas["rendimiento"]/100)**x
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=x,y=y,mode="lines+markers",name="Valor hipotético",line=dict(color=CYAN,width=4),
                                     fill="tozeroy",fillcolor="rgba(4,224,255,.10)",hovertemplate="Año %{x}<br>$%{y:,.0f} MXN<extra></extra>"))
            fig.add_hline(y=capital,line_dash="dot",line_color="#657FA7",annotation_text="Capital inicial")
            fig.update_layout(xaxis_title="Años",yaxis_title="MXN",showlegend=False)
            look(fig,330)
        with b:
            back=cartera["backtest"].get("5",{})
            if perfil=="Agresivo":
                st.markdown("#### Walk-forward 2022–2025")
                st.metric("CAGR histórico WF",formato(back["cagr"]))
                st.metric("Drawdown máximo WF",formato(back["max_drawdown"]))
                st.metric("CAGR SPY mismo periodo",formato(cartera["validacion"]["spy_cagr"]))
                st.caption("Métricas fuera de muestra de una estrategia reoptimizada por año; no son resultados históricos exactos de los pesos actuales fijos.")
            else:
                st.markdown("#### Datos históricos (estratégico)")
                st.metric("CAGR histórico 5A",formato(back["cagr"]))
                st.metric("Drawdown máximo 5A",formato(back["max_drawdown"]))
                st.metric("Exceso anualizado vs benchmark",formato(back["exceso_benchmark"]))
                st.caption("Backtest del portafolio estratégico del estudio; NO corresponde a la estrategia optimizada que se seleccione si es distinta.")
        if perfil == "Agresivo":
            st.markdown("#### Escenarios ilustrativos del notebook Refinitiv")
            cs=st.columns(3)
            for col,(etiqueta,tasa) in zip(cs,cartera["proyeccion_escenarios"].items()):
                with col:
                    with st.container(border=True):
                        st.markdown(f"**{etiqueta.upper()}**")
                        st.metric("Tasa supuesta anual",formato(tasa))
                        st.metric(f"Capital a {anos} años",moneda(capital*(1+tasa/100)**anos))
            st.caption("Escenarios de 6%, 9% y 12% tomados del notebook. No son simulaciones Monte Carlo ni rentabilidades estimadas específicamente para esta cartera.")
        else:
            st.markdown("#### Escenarios Monte Carlo — referencia del estudio a 5 años")
            esc=cartera["montecarlo"]
            cs=st.columns(3)
            for col,n in zip(cs,["Pesimista","Base","Optimista"]):
                e=esc[n]
                with col:
                    with st.container(border=True):
                        st.markdown(f"**{n.upper()}**")
                        st.metric("Mediana anual simulada",formato(e["rend_anual_mediana"]))
                        st.metric("Pérdida en 12 meses (modelo)",formato(e["prob_perdida_12m"]))
                        st.caption(f'VaR 95% a 12m: {formato(e["var_95_12m"])}')
            st.caption("Cifras precalculadas del informe original. No son probabilidades específicas de tu formulario ni se vuelven a simular al modificar capital o estrategia.")
    with t3:
        st.markdown("#### Momentum técnico de los activos")
        st.caption("Tendencias del informe al 7/oct/2026 para activos con cobertura; las señales del nuevo portafolio agresivo no se proporcionaron y se muestran como Sin dato. No son datos en tiempo real.")
        tecnica=df[["Ticker","Peso (%)","Tendencia","RSI (14)","Cruce"]]
        st.dataframe(tecnica,use_container_width=True,hide_index=True)
        st.caption("Los bonos mexicanos no tienen señal técnica individual en el archivo. «Sin dato» no significa tendencia lateral.")
        if clave=="Estrategico" and cartera.get("riesgo_pct"):
            st.markdown("#### Principales contribuciones al riesgo")
            riesgos=sorted(cartera["riesgo_pct"].items(),key=lambda p:p[1],reverse=True)[:8]
            fig=px.bar(x=[x[1] for x in riesgos][::-1],y=[x[0] for x in riesgos][::-1],orientation="h")
            fig.update_traces(marker_color=BLUE)
            fig.update_layout(xaxis_title="% de contribución al riesgo",yaxis_title="",showlegend=False)
            look(fig,310)
        else:
            st.caption("No hay desglose de contribución al riesgo documentado para esta estrategia.")

    html('''<div class="disclaimer"><b>Limitaciones del proyecto:</b> SCF 2022 estudia hogares estadounidenses; sus predicciones no están validadas para usuarios mexicanos. En el portafolio conservador, parte de la renta fija mexicana se aproximó con series/proxies de FRED que pueden subestimar correlaciones y volatilidad. No es una orden ni recomendación regulada para invertir.</div>''')
    if st.button("← Hacer un nuevo diagnóstico"):
        st.session_state.fase="cuestionario";st.rerun()


def laboratorio(bundle):
    html('''<div class="hero"><div class="pill">MODEL LAB • 6 ALGORITMOS</div>
    <h1>Detrás de cada <span style="color:#36E6FF">predicción.</span></h1>
    <p>Evaluación reproducible en test usando una única partición estratificada. Puedes contrastar la matriz de errores y el desempeño por perfil.</p></div>''')
    st.write("")
    met=bundle["metricas"]
    comp=pd.DataFrame([{"Modelo":n,"Accuracy":m["accuracy"]*100,"F1 macro":m["f1_macro"]*100,
                        "Recall conservador":m["recall_clases"][0]*100,"Recall moderado":m["recall_clases"][1]*100,
                        "Recall agresivo":m["recall_clases"][2]*100} for n,m in met.items()]).sort_values("Accuracy",ascending=False)
    c1,c2=st.columns([1.2,.8])
    with c1:
        fig=px.bar(comp,x="Modelo",y=["Accuracy","F1 macro"],barmode="group",color_discrete_sequence=[BLUE,CYAN])
        fig.update_layout(yaxis_title="%",legend_title="Métrica")
        look(fig,380)
    with c2:
        st.markdown("#### Resultados fuera de muestra")
        st.metric("Hogares de entrenamiento",f'{bundle["n_entrenamiento"]:,}')
        st.metric("Hogares de prueba",f'{bundle["n_prueba"]:,}')
        st.metric("Variables predictoras", "7")
    st.dataframe(comp.round(2),use_container_width=True,hide_index=True)
    st.markdown(f"#### Matriz de confusión — {nombre_modelo}")
    fig=px.imshow(np.array(met[nombre_modelo]["matriz"]),x=PERFILES,y=PERFILES,text_auto=True,
                  color_continuous_scale=[[0,"#0A1730"],[.5,"#0F5AA0"],[1,"#00DBFF"]],aspect="auto",
                  labels={"x":"Predicción","y":"Perfil real","color":"Casos"})
    look(fig,365)
    st.markdown("#### Las siete variables seleccionadas por información mutua")
    importancia={"Ingreso anual":.140832,"Cuenta de inversión":.105813,"Educación":.102421,
                 "Tiene acciones":.087257,"Tiene fondos":.054112,"Horizonte de planeación":.042431,
                 "Conocimiento financiero":.039462}
    imp=pd.DataFrame({"Variable":list(importancia),"Información mutua":list(importancia.values())}).sort_values("Información mutua")
    fig=px.bar(imp,x="Información mutua",y="Variable",orientation="h")
    fig.update_traces(marker_color=CYAN)
    look(fig,300)
    st.caption("Puntajes del notebook original. Miden asociación informativa con la etiqueta, no importancia causal ni pesos de cada modelo.")
    st.info("SVM y XGBoost aquí se vuelven a entrenar con el conjunto SCF del proyecto. Sus métricas pueden variar ligeramente respecto a las imágenes originales: el escalado de esta app se ajusta solo con entrenamiento, evitando fuga de información hacia test.")


if pagina == "Mi experiencia" and st.session_state.fase == "cuestionario":
    cuestionario()
else:
    try:
        with st.spinner("Preparando modelos reales con SCF 2022 (la primera vez se descarga y entrena la base)..."):
            bundle=obtener_modelos()
    except Exception as ex:
        pantalla_error(ex)
    else:
        if pagina=="Laboratorio de modelos": laboratorio(bundle)
        elif "respuestas" in st.session_state: mostrar_resultado(bundle)
        else: cuestionario()
