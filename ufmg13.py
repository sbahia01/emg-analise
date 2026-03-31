import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt
from scipy.integrate import simpson

# ==========================================================
# 1. IDENTIDADE VISUAL - HUB ACADEMICA
# ==========================================================
HUB_NAVY = "#001a33"
HUB_BLUE = "#1a73e8"
HUB_LIGHT = "#f8f9fa"
WHITE = "#ffffff"

st.set_page_config(page_title="EMG Expert | Hub Academica", layout="wide")

# ==========================================================
# 2. DICIONÁRIO DE TRADUÇÕES
# ==========================================================
LANGUAGES = {
    "PORTUGUÊS (BRASILEIRO)": {
        "title": "EMGExpert — Ciência Encontra a Prática",
        "upload_label": "Carregar arquivo (.slk ou .csv)",
        "info_msg": "🖱️ Arraste no gráfico para analisar.",
        "report_title": "RELATÓRIO TÉCNICO",
        "onset": "ONSET (INÍCIO)",
        "peak": "PICO MÁXIMO",
        "area": "ÁREA (INTEGRAL)",
        "delay_title": "Diferença de Sincronismo",
        "wait_msg": "Aguardando upload do arquivo..."
    },
    "ENGLISH": {
        "title": "EMGExpert — Science Meets Practice",
        "upload_label": "Upload file (.slk or .csv)",
        "info_msg": "🖱️ Drag on the chart to analyze.",
        "report_title": "TECHNICAL REPORT",
        "onset": "ONSET",
        "peak": "PEAK AMPLITUDE",
        "area": "AREA (INTEGRAL)",
        "delay_title": "Synchronization Delay",
        "wait_msg": "Waiting for file upload..."
    },
    "ESPAÑOL": {
        "title": "EMGExpert — La Ciencia se Encuentra con la Práctica",
        "upload_label": "Cargar archivo (.slk o .csv)",
        "info_msg": "🖱️ Arrastre en el gráfico para analizar.",
        "report_title": "INFORME TÉCNICO",
        "onset": "COMIENZO (ONSET)",
        "peak": "PICO MÁXIMO",
        "area": "ÁREA (INTEGRAL)",
        "delay_title": "Diferencia de Sincronismo",
        "wait_msg": "Esperando carga de archivo..."
    },
    "CHINESE (SIMPLIFIED)": {
        "title": "EMGExpert — 科学与实践的结合",
        "upload_label": "上传文件 (.slk 或 .csv)",
        "info_msg": "🖱️ 在图表上拖动以进行分析。",
        "report_title": "技术报告",
        "onset": "起始点 (Onset)",
        "peak": "最大峰值",
        "area": "面积 (积分)",
        "delay_title": "同步延迟",
        "wait_msg": "等待文件上传..."
    }
}

# CSS Personalizado
st.markdown(f"""
    <style>
    .stApp {{ background-color: {WHITE}; color: {HUB_NAVY}; }}
    [data-testid="stSidebar"] {{ background-color: {HUB_LIGHT}; border-right: 1px solid #e0e0e0; }}
    .report-card {{ 
        border-left: 5px solid {HUB_NAVY}; padding: 15px; 
        background-color: {HUB_LIGHT}; border-radius: 4px;
        color: {HUB_NAVY}; font-family: 'Inter', sans-serif;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05); margin-top: 10px;
    }}
    h1, h2, h3 {{ color: {HUB_NAVY} !important; font-weight: 700 !important; }}
    </style>
    """, unsafe_allow_html=True)

# Interface de Cabeçalho
col_t, col_l = st.columns([4, 1])
with col_l:
    selected_lang = st.selectbox("🌐 Language", list(LANGUAGES.keys()))
    t = LANGUAGES[selected_lang]
with col_t:
    st.title(t["title"])

# ==========================================================
# 3. LÓGICA TÉCNICA (FILTROS E ANALISES)
# ==========================================================

def process_emg(data, fs=2000):
    nyq = 0.5 * fs
    b, a = butter(4, [6/nyq, 500/nyq], btype='bandpass')
    filt = filtfilt(b, a, data)
    rect = np.abs(filt - np.mean(filt))
    window = int(fs * 0.01)
    return np.sqrt(np.convolve(rect**2, np.ones(window)/window, mode='same'))

def get_onset(rms_seg, full_rms, fs=2000):
    baseline = full_rms[:400]
    thresh = np.mean(baseline) + (3 * np.std(baseline))
    check = int(0.02 * fs)
    for i in range(len(rms_seg) - check):
        if np.all(rms_seg[i : i + check] >= thresh):
            return i, thresh
    return None, thresh

# ==========================================================
# 4. EXECUÇÃO
# ==========================================================

file = st.sidebar.file_uploader(t["upload_label"], type=["slk", "csv"])

if file:
    # Leitura SYLK Robusta
    content = file.getvalue().decode("utf-8", errors="ignore")
    data_map = {}
    names = {4: "CH 1", 5: "CH 2"}
    for line in content.splitlines():
        if line.startswith('C;'):
            p = line.split(';')
            try:
                r, c = int(p[1][1:]), int(p[2][1:])
                val = float(p[3][1:].replace('"','')) if p[3].startswith('K') else float(p[3].replace('"',''))
                if r not in data_map: data_map[r] = {}
                data_map[r][c] = val
                if r == 4 and c in [4, 5]: names[c] = p[3][1:].replace('"','')
            except: continue
    
    df_raw = pd.DataFrame.from_dict(data_map, orient='index').sort_index()
    df = pd.DataFrame({'time': df_raw[1].values, 'CH1': df_raw[4].values, 'CH2': df_raw[5].values}).dropna().iloc[5:]
    
    fs, onsets = 2000, {}
    cols = st.columns(2)

    for i, (ch, label) in enumerate(zip(['CH1', 'CH2'], [names[4], names[5]])):
        with cols[i]:
            st.subheader(label)
            rms = process_emg(df[ch].values)
            
            fig = go.Figure(go.Scatter(x=df['time'], y=rms, line=dict(color=HUB_NAVY, width=1.2)))
            fig.update_layout(
                height=380, dragmode='select', selectdirection='h',
                plot_bgcolor=WHITE, margin=dict(l=0, r=0, t=10, b=0),
                newshape=dict(line=dict(color=HUB_BLUE, width=2), fillcolor=HUB_BLUE, opacity=0.2),
                xaxis=dict(showgrid=True, gridcolor='#f0f0f0'), yaxis=dict(showgrid=True, gridcolor='#f0f0f0')
            )
            
            sel = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key=f"p_{ch}_{selected_lang}")

            if sel and "selection" in sel and "box" in sel["selection"] and len(sel["selection"]["box"]) > 0:
                t1, t2 = sel["selection"]["box"][0]["x"][0], sel["selection"]["box"][0]["x"][1]
                mask = (df['time'] >= t1) & (df['time'] <= t2)
                s_t, s_r = df['time'][mask].values, rms[mask]
                
                if len(s_r) > 10:
                    idx, thr = get_onset(s_r, rms)
                    v_max, area = np.max(s_r), simpson(s_r, dx=1/fs)
                    
                    st.markdown(f"""
                    <div class="report-card">
                        <small>{t['report_title']}</small><br>
                        <b>{t['onset']}:</b> {s_t[idx]:.4f} s<br>
                        <b>{t['peak']}:</b> {v_max:.2f} µV<br>
                        <b>{t['area']}:</b> {area:.4f} µV.s
                    </div>
                    """, unsafe_allow_html=True)
                    if idx is not None: onsets[i] = s_t[idx]
            else:
                st.info(t["info_msg"])

    if len(onsets) == 2:
        st.success(f"### ⏱️ {t['delay_title']}: {abs(onsets[0]-onsets[1])*1000:.2f} ms")
else:
    st.info(t["wait_msg"])
