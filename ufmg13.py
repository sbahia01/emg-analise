import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt
from scipy.integrate import simpson

# ==========================================================
# 1. CONFIGURAÇÃO VISUAL (BRANCO E PRETO)
# ==========================================================
st.set_page_config(page_title="Análise EMG UFMG", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: white !important; color: black !important; }
    h1, h2, h3, p, span, label, .stMarkdown { color: black !important; font-family: sans-serif; }
    .report-box { 
        border: 1px solid #000; 
        padding: 15px; 
        font-family: monospace; 
        margin-top: 10px;
        background-color: #fff;
        color: #000;
        line-height: 1.5;
    }
    hr { border: 0.5px solid #000; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================================
# 2. FUNÇÕES TÉCNICAS (CONSOLIDADO UFMG.PY)
# ==========================================================

def read_slk_file(file):
    data = {}
    muscle_names = {}
    content = file.getvalue().decode("utf-8", errors="ignore")
    for line in content.splitlines():
        if line.startswith('C;'):
            parts = line.strip().split(';')
            try:
                row, col = int(parts[1][1:]), int(parts[2][1:])
                val = parts[3][1:].strip('"') if parts[3].startswith('K') else parts[3]
                if row not in data: data[row] = {}
                data[row][col] = val
                if row == 4 and col in [4, 5]: muscle_names[col] = val
            except: continue
    df = pd.DataFrame(data).T.sort_index()
    df = df.iloc[4:, [0, 3, 4]].reset_index(drop=True).astype(float)
    df.columns = ['time', 'CH1', 'CH2']
    return df, muscle_names

def process_emg_segment(segment_signal, fs=2000):
    """Processamento completo do sinal conforme UFMG.py"""
    # Filtro Butterworth 4ª ordem (6-500Hz)
    nyq = 0.5 * fs
    b, a = butter(4, [6/nyq, 500/nyq], btype='bandpass')
    filt = filtfilt(b, a, segment_signal)
    
    # Retificação
    rect = np.abs(filt - np.mean(filt))
    
    # Moving RMS de 10ms (janela de 20 amostras para 2000Hz)
    window = int(fs * 0.01)
    rms = np.sqrt(np.convolve(rect**2, np.ones(window)/window, mode='same'))
    return rms

def detect_onset(rms_signal, fs=2000):
    """Detecção de Onset: Baseline (primeiros 400 pontos) + 3 Desvios Padrão"""
    if len(rms_signal) < 400:
        return None, 0
    base_mean = np.mean(rms_signal[:400])
    base_std = np.std(rms_signal[:400])
    thresh = base_mean + (3 * base_std)
    
    # Busca por 20ms (40 amostras) sustentados acima do threshold
    for i in range(len(rms_signal) - 40):
        if np.all(rms_signal[i : i + 40] >= thresh):
            return i, thresh
    return None, thresh

# ==========================================================
# 3. INTERFACE E LÓGICA DE SELEÇÃO DINÂMICA
# ==========================================================
st.title("Análise de EMG - Protocolo UFMG")
st.write("---")
st.info("🖱️ **Instrução:** Clique e arraste o mouse lateralmente no gráfico para selecionar o trecho da contração.")

uploaded_file = st.sidebar.file_uploader("Upload do Arquivo (.slk ou .csv)", type=["slk", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith('.slk'):
        df_total, names = read_slk_file(uploaded_file)
        labels = [names.get(4, "Músculo 1"), names.get(5, "Músculo 2")]
    else:
        df_total = pd.read_csv(uploaded_file)
        df_total.columns = ['time', 'CH1', 'CH2']
        labels = ["Canal 1", "Canal 2"]

    fs = 2000
    onsets_results = {}
    cols_ui = st.columns(2)

    for i, (ch_key, label, col_ui) in enumerate(zip(['CH1', 'CH2'], labels, cols_ui)):
        with col_ui:
            st.subheader(label)
            
            # 1. Gerar RMS do sinal completo apenas para visualização no gráfico
            full_rms_view = process_emg_segment(df_total[ch_key].values, fs)
            
            fig = go.Figure(go.Scatter(x=df_total['time'], y=full_rms_view, line=dict(color='black', width=1)))
            fig.update_layout(
                height=350, margin=dict(l=0,r=0,t=
