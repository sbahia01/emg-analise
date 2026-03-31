import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt
from scipy.integrate import simpson
import io

# ==========================================================
# 1. CONFIGURAÇÃO DE AMBIENTE E ESTILO
# ==========================================================
st.set_page_config(page_title="Análise EMG UFMG - Professional", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: white; color: black; }
    .report-card { 
        border: 2px solid #000; padding: 20px; 
        background-color: #ffffff; color: #000;
        font-family: 'Courier New', Courier, monospace;
        box-shadow: 5px 5px 0px #eee;
        margin-bottom: 20px;
    }
    .metric-title { font-weight: bold; border-bottom: 1px solid #000; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================================
# 2. MOTOR DE PROCESSAMENTO TÉCNICO (PROTOCOLO UFMG)
# ==========================================================

def butter_bandpass_filter(data, lowcut=6, highcut=500, fs=2000, order=4):
    """Filtro Butterworth de 4ª ordem conforme protocolo UFMG."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    # Proteção: só filtra se houver dados suficientes
    if len(data) <= (max(len(a), len(b)) * 3):
        return data
    y = filtfilt(b, a, data)
    return y

def calculate_rms(data, fs=2000, window_ms=10):
    """Cálculo de RMS com janela móvel de 10ms."""
    rectified = np.abs(data - np.mean(data))
    window_size = int(fs * (window_ms / 1000))
    if window_size < 1: window_size = 1
    # Convolução para RMS estável
    rms = np.sqrt(np.convolve(rectified**2, np.ones(window_size)/window_size, mode='same'))
    return rms

def find_onset_threshold(rms_segment, full_rms_signal, fs=2000):
    """Detecta Onset: Primeiro ponto que fica 3 STDs acima da média do repouso por 20ms."""
    # Baseline: Primeiros 400 pontos (200ms a 2000Hz) do sinal total
    baseline = full_rms_signal[:400]
    mean_base = np.mean(baseline)
    std_base = np.std(baseline)
    threshold = mean_base + (3 * std_base)
    
    samples_to_check = int(0.02 * fs) # 20ms
    for i in range(len(rms_segment) - samples_to_check):
        if np.all(rms_segment[i : i + samples_to_check] >= threshold):
            return i, threshold
    return None, threshold

# ==========================================================
# 3. LEITURA DE DADOS (SLK / CSV) - ALTA ROBUSTEZ
# ==========================================================

def load_data(file):
    """Lê o arquivo garantindo a extração correta das colunas de tempo e canais."""
    try:
        content = file.getvalue().decode("utf-8", errors="ignore")
        lines = content.splitlines()
        
        data_rows = []
        labels = {4: "Canal 1", 5: "Canal 2"}
        
        for line in lines:
            if line.startswith('C;'):
                parts = line.split(';')
                try:
                    # Formato SYLK: C;Y(row);X(col);K(value)
                    r = int(parts[1][1:])
                    c = int(parts[2][1:])
                    v_str = parts[3]
                    if v_str.startswith('K'):
                        val = float(v_str[1:].replace('"', ''))
                    else:
                        val = float(v_str.replace('"', ''))
                    
                    data_rows.append({'row': r, 'col': c, 'val': val})
                    
                    # Tenta capturar nomes de músculos se existirem na linha 4
                    if r == 4 and c in [4, 5] and v_str.startswith('K'):
                        labels[c] = v_str[1:].replace('"', '')
                except:
                    continue
        
        if not data_rows:
            return None, None
