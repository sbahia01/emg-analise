import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt
from scipy.integrate import simpson

# 1. CORES HUB ACADEMICA
HUB_NAVY = "#001a33"
HUB_BLUE = "#1a73e8"
HUB_LIGHT = "#f8f9fa"

st.set_page_config(page_title="EMGExpert | Hub Academica", layout="wide")

# 2. TRADUÇÕES
LANGS = {
    "PORTUGUÊS (BRASILEIRO)": {
        "title": "EMGExpert — Ciência Encontra a Prática",
        "upload": "Carregar arquivo (.slk ou .csv)",
        "info": "🖱️ Arraste no gráfico para analisar.",
        "rep": "RELATÓRIO TÉCNICO",
        "ons": "ONSET", "peak": "PICO", "area": "ÁREA",
        "sync": "Diferença de Sincronismo", "wait": "Aguarde o upload..."
    },
    "ENGLISH": {
        "title": "EMGExpert — Science Meets Practice",
        "upload": "Upload file",
        "info": "🖱️ Drag to analyze.",
        "rep": "TECHNICAL REPORT",
        "ons": "ONSET", "peak": "PEAK", "area": "AREA",
        "sync": "Delay", "wait": "Waiting..."
    }
}

# 3. CSS ESTÁVEL (CORRIGINDO O BOTÃO E CORES)
st.markdown(f"""
    <style>
    /* Fundo e Barra Lateral */
    .stApp {{ background-color: white; }}
    [data-testid="stSidebar"] {{ background-color: {HUB_NAVY}; }}
    [data-testid="stSidebar"] * {{ color: white !important; }}
    
    /* BOTÃO DE UPLOAD (Destaque total) */
    div[data-testid="stFileUploadDropzone"] {{
        background-color: white !important;
        border: 2px solid {HUB_BLUE} !important;
        color: {HUB_NAVY} !important;
    }}

    /* Títulos */
    h1, h2, h3 {{ color: {HUB_NAVY} !important; }}
    
    /* Cartão de Relatório */
    .report-card {{
        border-left: 5px solid {HUB_BLUE};
        background-color: {HUB_LIGHT};
        padding: 15px;
        color: {HUB_NAVY};
        border-radius: 5px;
        margin-top: 10px;
    }}
    </style>
    """, unsafe_allow_html=True)

# 4. FUNÇÕES DE PROCESSAMENTO
def butter_filter(data):
    fs = 2000
    nyq = 0.5 * fs
    b, a = butter(4, [6/nyq, 500/nyq], btype='band')
    y = filtfilt(b, a, data)
    return np.abs(y - np.mean(y))

def parse_sylk(file):
    try:
        content = file.getvalue().decode("utf-8", errors="ignore")
        data_map = {}
        for line in content.splitlines():
            if line.startswith('C;'):
                p = line.split(';')
                try:
                    r, c = int(p[1][1:]), int(p[2][1:])
                    v = p[3]
                    val = float(v[1:].replace('"','')) if v.startswith('K') else float(v.replace('"',''))
                    if r not in data_map: data_map[r] = {}
                    data_map[r][c] = val
                except: continue
        df_r = pd.DataFrame.from_dict(data_map, orient='index').sort_index()
        df = pd.DataFrame({'time': df_r[1].values, 'CH1': df_r[4].values, 'CH2': df_r[5].values}).dropna().iloc[5:]
        return df
    except: return None

# 5. INTERFACE
header_col1, header_col2 = st.columns([4, 1])
with header_col2:
    sel_lang = st.selectbox("🌐 Language", list(LANGS.keys()))
    tr = LANGS[sel_lang]
with header_col1:
    st.title(tr["title"])

up_file = st.sidebar.file_uploader(tr["upload"], type=["slk", "csv"])

if up_file:
    df = parse_sylk(up_file)
    if df is not None:
        onsets = {}
        cols = st.columns(2)
        for i, ch in enumerate(['CH1', 'CH2']):
            with cols[i]:
                st.subheader(f"Canal {i+1}")
                rect = butter_filter(df[ch].values)
                
                fig = go.Figure(go.Scatter(x=df['time'], y=rect, line=dict(color=HUB_NAVY, width=1)))
                fig.update_layout(height=350, dragmode='select', selectdirection='h', plot_bgcolor='white', margin=dict(l=0,r=0,t=0,b=0))
                
                sel = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key=f"plot_{ch}_{sel_lang}")

                if sel and "selection" in sel and "box" in sel["selection"] and len(sel["selection"]["box"]) > 0:
                    t1, t2 = sel["selection"]["box"][0]["x"][0], sel["selection"]["box"][0]["x"][1]
                    mask = (df['time'] >= t1) & (df['time'] <= t2)
                    s_t, s_r = df['time'][mask].values, rect[mask]
                    
                    if len(s_r) > 10:
                        v_max = np.max(s_r)
                        area = simpson(s_r, dx=1/2000)
                        
                        st.markdown(f"""
                        <div class="report-card">
                            <b>{tr['rep']}</b><br>
                            • {tr['peak']}: {v_max:.2f} µV<br>
                            • {tr['area']}: {area:.4f} µV.s
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info(tr["info"])
else:
    st.info(tr["wait"])
