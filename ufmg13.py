import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt
from scipy.integrate import simpson
import io

# ==========================================================
# 1. DICIONÁRIO DE TRADUÇÕES (SISTEMA MULTI-IDIOMA)
# ==========================================================
LANGUAGES = {
    "PORTUGUÊS (BRASILEIRO)": {
        "title": "Análise de EMG - Protocolo UFMG",
        "upload_label": "Carregar arquivo (.slk ou .csv)",
        "sidebar_cfg": "Configurações",
        "info_msg": "🖱️ Clique e arraste no gráfico para analisar.",
        "report_title": "RELATÓRIO TÉCNICO",
        "interval": "Intervalo Selecionado",
        "onset": "ONSET",
        "peak": "PICO MÁXIMO",
        "t_peak": "TEMPO DO PICO",
        "area": "ÁREA (INTEGRAL)",
        "thresh": "Threshold de detecção",
        "delay_title": "Diferença de Sincronismo (Delay)",
        "wait_msg": "Aguardando upload do arquivo..."
    },
    "ENGLISH": {
        "title": "EMG Analysis - UFMG Protocol",
        "upload_label": "Upload file (.slk or .csv)",
        "sidebar_cfg": "Settings",
        "info_msg": "🖱️ Click and drag on the chart to analyze.",
        "report_title": "TECHNICAL REPORT",
        "interval": "Selected Range",
        "onset": "ONSET",
        "peak": "PEAK AMPLITUDE",
        "t_peak": "PEAK TIME",
        "area": "AREA (INTEGRAL)",
        "thresh": "Detection Threshold",
        "delay_title": "Synchronization Difference (Delay)",
        "wait_msg": "Waiting for file upload..."
    },
    "ESPAÑOL": {
        "title": "Análisis de EMG - Protocolo UFMG",
        "upload_label": "Cargar archivo (.slk o .csv)",
        "sidebar_cfg": "Configuraciones",
        "info_msg": "🖱️ Haga clic y arrastre en el gráfico para analizar.",
        "report_title": "INFORME TÉCNICO",
        "interval": "Intervalo Seleccionado",
        "onset": "COMIENZO (ONSET)",
        "peak": "PICO MÁXIMO",
        "t_peak": "TIEMPO DEL PICO",
        "area": "ÁREA (INTEGRAL)",
        "thresh": "Umbral de detección",
        "delay_title": "Diferencia de Sincronismo (Delay)",
        "wait_msg": "Esperando carga de archivo..."
    },
    "CHINESE (SIMPLIFIED)": {
        "title": "肌电图分析 - UFMG 协议",
        "upload_label": "上传文件 (.slk 或 .csv)",
        "sidebar_cfg": "设置",
        "info_msg": "🖱️ 在图表上点击并拖动以进行分析。",
        "report_title": "技术报告",
        "interval": "选定范围",
        "onset": "起始点 (Onset)",
        "peak": "最大峰值",
        "t_peak": "峰值时间",
        "area": "面积 (积分)",
        "thresh": "检测阈值",
        "delay_title": "同步差异 (延迟)",
        "wait_msg": "等待文件上传..."
    }
}

# ==========================================================
# 2. CONFIGURAÇÕES GERAIS
# ==========================================================
st.set_page_config(page_title="EMG UFMG Multi-Lang", layout="wide")

# Seletor de Idioma no topo superior direito (usando colunas)
col_title, col_lang = st.columns([4, 1])

with col_lang:
    selected_lang = st.selectbox("🌐 Language/Idioma", list(LANGUAGES.keys()))
    t = LANGUAGES[selected_lang] # Atalho para o dicionário de tradução

with col_title:
    st.title(t["title"])

st.markdown("""
    <style>
    .stApp { background-color: #ffffff; color: #000000; }
    .report-box { 
        border: 2px solid #000000; padding: 15px; 
        background-color: #f9f9f9; color: #000000;
        font-family: 'Courier New', Courier, monospace;
        margin-top: 10px; box-shadow: 4px 4px 0px #cccccc;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================================
# 3. FUNÇÕES TÉCNICAS (MATEMÁTICA NÃO MUDA COM IDIOMA)
# ==========================================================

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return b, a

def apply_emg_filter(data, fs=2000):
    if len(data) < 15: return data
    b, a = butter_bandpass(6, 500, fs, order=4)
    y = filtfilt(b, a, data)
    return np.abs(y - np.mean(y))

def calculate_rms_signal(rectified_data, fs=2000):
    window_size = int(fs * 0.01)
    return np.sqrt(np.convolve(rectified_data**2, np.ones(window_size)/window_size, mode='same'))

def detect_onset(rms_segment, full_rms, fs=2000):
    baseline = full_rms[:400]
    thresh = np.mean(baseline) + (3 * np.std(baseline))
    check_samples = int(0.020 * fs) 
    for i in range(len(rms_segment) - check_samples):
        if np.all(rms_segment[i : i + check_samples] >= thresh):
            return i, thresh
    return None, thresh

def parse_slk_file(uploaded_file):
    try:
        content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        data_map = {}
        ch_names = {4: "CH1", 5: "CH2"}
        for line in content.splitlines():
            if line.startswith('C;'):
                parts = line.split(';')
                try:
                    r, c = int(parts[1][1:]), int(parts[2][1:])
                    v_part = parts[3]
                    val = float(v_part[1:].replace('"', '')) if v_part.startswith('K') else float(v_part.replace('"', ''))
                    if r not in data_map: data_map[r] = {}
                    data_map[r][c] = val
                    if r == 4 and c in [4, 5] and v_part.startswith('K'):
                        ch_names[c] = v_part[1:].replace('"', '')
