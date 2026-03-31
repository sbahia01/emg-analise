import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt
from scipy.integrate import simpson

# ==========================================================
# 1. DEFINIÇÃO DA PALETA HUB ACADEMICA
# ==========================================================
HUB_NAVY = "#001a33"   # Azul Marinho (Barra lateral)
HUB_BLUE = "#1a73e8"   # Azul Vibrante (Botões e Destaques)
HUB_BG = "#f0f2f6"     # Cinza de fundo para dar contraste
WHITE = "#ffffff"

st.set_page_config(page_title="EMGExpert | Hub Academica", layout="wide")

# ==========================================================
# 2. DICIONÁRIO DE TRADUÇÕES
# ==========================================================
LANGS = {
    "PORTUGUÊS (BRASILEIRO)": {
        "title": "EMGExpert — Ciência Encontra a Prática",
        "upload": "Selecione o arquivo (.slk ou .csv)",
        "info": "🖱️ Arraste no gráfico para analisar a contração.",
        "rep": "RELATÓRIO TÉCNICO",
        "ons": "ONSET (INÍCIO)",
        "peak": "PICO MÁXIMO",
        "area": "ÁREA (INTEGRAL)",
        "sync": "Diferença de Sincronismo (Delay)",
        "wait": "Aguardando upload do arquivo...",
        "thresh": "Threshold de detecção"
    },
    "ENGLISH": {
        "title": "EMGExpert — Science Meets Practice",
        "upload": "Choose file (.slk or .csv)",
        "info": "🖱️ Drag on the chart to analyze.",
        "rep": "TECHNICAL REPORT",
        "ons": "ONSET",
        "peak": "PEAK AMPLITUDE",
        "area": "AREA",
        "sync": "Sync Delay",
        "wait": "Waiting for file...",
        "thresh": "Threshold"
    }
}

# ==========================================================
# 3. CSS AVANÇADO (RESOLVENDO O PROBLEMA DE CORES)
# ==========================================================
st.markdown(f"""
    <style>
    /* Fundo de toda a aplicação */
    .stApp {{
        background-color: {HUB_BG} !important;
    }}
    
    /* Barra Lateral - Navy Blue */
    [data-testid="stSidebar"] {{
        background-color: {HUB_NAVY} !important;
    }}
    [data-testid="stSidebar"] * {{
        color: {WHITE} !important;
    }}
    
    /* BOTÃO DE UPLOAD - Resolvendo a invisibilidade */
    section[data-testid="stFileUploadDropzone"] {{
        background-color: {WHITE} !important;
        border: 2px dashed {HUB_BLUE} !important;
        border-radius: 10px !important;
        padding: 20px !important;
    }}
    section[data-testid="stFileUploadDropzone"] i {{
        color: {HUB_BLUE} !important;
    }}
    
    /* Títulos e Subtítulos */
    h1, h2, h3, .stMarkdown p {{
        color: {HUB_NAVY} !important;
        font-family: 'Inter', sans-serif;
    }}

    /* Cartões de Relatório - Brancos sobre o fundo cinza */
    .report-card {{
        background-color: {WHITE} !important;
        border-top: 5px solid {HUB_BLUE} !important;
        padding: 25px !important;
        border-radius: 10px !important;
        color: {HUB_NAVY} !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
        margin-bottom: 25px !important;
    }}
    
    /* Estilização dos Selectboxes */
    .stSelectbox div[data-baseweb="select"] {{
        background-color: {WHITE} !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# ==========================================================
# 4. MOTOR TÉCNICO
# ==========================================================

def butter_bandpass_filter(data, fs=2000):
    nyq = 0.5 * fs
    b, a = butter(4, [6/nyq, 500/nyq], btype='band')
    if len(data) <= 12: return data
    return filtfilt(b, a, data)

def calculate_rms(data, fs=2000):
    rectified = np.abs(data - np.mean(data))
    window = int(fs * 0.01)
    return np.sqrt(np.convolve(rectified**2, np.ones(window)/window, mode='same'))

def parse_sylk(file):
    try:
        content = file.getvalue().decode("utf-8", errors="ignore")
        data_map = {}
        ch_names = {4: "Canal 1", 5: "Canal 2"}
        for line in content.splitlines():
            if line.startswith('C;'):
                p = line.split(';')
                try:
                    r, c = int(p[1][1:]), int(p[2][1:])
                    val_raw = p[3]
                    val = float(val_raw[1:].replace('"','')) if val_raw.startswith('K') else float(val_raw.replace('"',''))
                    if r not in data_map: data_map[r] = {}
                    data_map[r][c] = val
                    if r == 4 and c in [4, 5]: ch_names[c] = val_raw[1:].replace('"','')
                except: continue
        df_raw = pd.DataFrame.from_dict(data_map, orient='index').sort_index()
        df = pd.DataFrame({'time': df_raw[1].values, 'CH1': df_raw[4].values, 'CH2': df_raw[5].values}).dropna().iloc[5:]
        return df, [ch_names[4], ch_names[5]]
    except: return None, None

# ==========================================================
# 5. INTERFACE
# ==========================================================

header_col1, header_col2 = st.columns([4, 1])
with header_col2:
    sel_lang = st.selectbox("🌐 Language", list(LANGS.keys()))
    tr = LANGS[sel_lang]
with header_col1:
    st.title(tr["title"])

uploaded_file = st.sidebar.file_uploader(tr["upload"], type=["slk", "csv"])

if uploaded_file:
    df_emg, labels = parse_sylk(uploaded_file)
    if df_emg is not None:
        fs = 2000
        onsets_results = {}
        ui_cols = st.columns(2)

        for i, (ch_col, name) in enumerate(zip(['CH1', 'CH2'], labels)):
            with ui_cols[i]:
                st.subheader(name)
                filt = butter_bandpass_filter(df_emg[ch_col].values)
                rms_signal = calculate_rms(filt)
                
                fig = go.Figure(go.Scatter(x=df_emg['time'], y=rms_signal, line=dict(color=HUB_NAVY, width=1.5)))
                fig.update_layout(
                    height=400, margin=dict(l=10, r=10, t=10, b=10),
                    dragmode='select', selectdirection='h',
                    plot_bgcolor=WHITE, paper_bgcolor='rgba(0,0,0,0)',
                    newshape=dict(line=dict(color=HUB_BLUE, width=2), fillcolor=HUB_BLUE, opacity=0.3),
                    xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
                    yaxis=dict(showgrid=True, gridcolor='#f0f0f0')
                )
                
                sel = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key=f"p_{ch_col}_{sel_lang}")

                if sel and "selection" in sel and "box" in sel["selection"] and len(sel["selection"]["box"]) > 0:
                    t1, t2 = sel["selection"]["box"][0]["x"][0], sel["selection"]["box"][0]["x"][1]
                    mask = (df_emg['time'] >= t1) & (df_emg['time'] <= t2)
                    st_time, st_rms = df_emg['time'][mask].values, rms_signal[mask]
                    
                    if len(st_rms) > 10:
                        baseline = rms_signal[:400]
                        thr = np.mean(baseline) + (3 * np.std(baseline))
                        idx = next((j for j in range(len(st_rms)-40) if np.all(st_rms[j:j+40] >= thr)), None)
                        
                        v_max = np.max(st_rms)
                        v_area = simpson(st_rms, dx=1/fs)
                        
                        st.markdown(f"""
                        <div
