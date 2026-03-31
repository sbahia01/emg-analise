import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt
from scipy.integrate import simpson

# ==========================================================
# 1. CONFIGURAÇÃO VISUAL
# ==========================================================
st.set_page_config(page_title="Análise EMG UFMG", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: white !important; color: black !important; }
    h1, h2, h3, p, span, label, .stMarkdown { color: black !important; font-family: sans-serif; }
    .report-box { 
        border: 2px solid #000; 
        padding: 15px; 
        font-family: 'Courier New', Courier, monospace; 
        margin-top: 10px;
        background-color: #f9f9f9;
        color: #000;
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================================
# 2. FUNÇÕES TÉCNICAS (Protocolo UFMG.py)
# ==========================================================

@st.cache_data
def load_and_process_file(file_bytes, file_name):
    """Lê o arquivo, aplica filtros e calcula RMS total"""
    fs = 2000
    
    # Leitura dos dados
    if file_name.endswith('.slk'):
        data = {}
        names = {}
        content = file_bytes.decode("utf-8", errors="ignore")
        for line in content.splitlines():
            if line.startswith('C;'):
                parts = line.strip().split(';')
                try:
                    row, col = int(parts[1][1:]), int(parts[2][1:])
                    val = parts[3][1:].strip('"') if parts[3].startswith('K') else parts[3]
                    if row not in data: data[row] = {}
                    data[row][col] = val
                    if row == 4 and col in [4, 5]: names[col] = val
                except: continue
        df = pd.DataFrame(data).T.sort_index()
        df = df.iloc[4:, [0, 3, 4]].reset_index(drop=True).astype(float)
        m_labels = [names.get(4, "Músculo 1"), names.get(5, "Músculo 2")]
    else:
        # Suporte a CSV
        df = pd.read_csv(io.BytesIO(file_bytes))
        if len(df.columns) > 3: df = df.drop(columns=df.columns[[1, 2]])
        df = df.iloc[3:].reset_index(drop=True).astype(float)
        m_labels = ["Canal 1", "Canal 2"]

    df.columns = ['time', 'CH1', 'CH2']
    
    # Processamento EMG (Filtro 6-500Hz, Retificação, RMS 10ms)
    nyq = 0.5 * fs
    b, a = butter(4, [6/nyq, 500/nyq], btype='bandpass')
    
    processed_data = {'time': df['time'].values}
    baselines = {}
    
    for ch in ['CH1', 'CH2']:
        sig = df[ch].values
        # Filtro e Detrend
        filt = filtfilt(b, a, sig)
        detrended = filt - np.mean(filt)
        rectified = np.abs(detrended)
        
        # RMS (10ms)
        window = int(fs * 0.01)
        rms = np.sqrt(np.convolve(rectified**2, np.ones(window)/window, mode='same'))
        processed_data[ch] = rms
        
        # Baseline Fixa (0 a 200ms do arquivo original)
        base_samples = int(fs * 0.2)
        base_mean = np.mean(rms[:base_samples])
        base_std = np.std(rms[:base_samples])
        threshold = base_mean + (3 * base_std)
        baselines[ch] = {'mean': base_mean, 'std': base_std, 'thresh': threshold}
        
    return processed_data, m_labels, baselines, fs

def detect_onset(window_signal, threshold, fs=2000):
    """Encontra o Onset dentro da janela selecionada (mantido > threshold por 20ms)"""
    min_samples = int(0.020 * fs) # 20 ms
    for i in range(len(window_signal)):
        if window_signal[i] >= threshold:
            end_check = min(i + min_samples, len(window_signal))
            if np.all(window_signal[i:end_check] >= threshold):
                return i
    return None

# ==========================================================
# 3. INTERFACE DE USUÁRIO
# ==========================================================
st.title("Análise de EMG - Protocolo UFMG")
st.write("---")

import io
uploaded_file = st.sidebar.file_uploader("Carregar arquivo (.slk ou .csv)", type=["slk", "csv"])

if uploaded_file:
    # 1. Carrega e processa o arquivo inteiro de uma vez (Cache)
    data, labels, baselines, fs = load_and_process_file(uploaded_file.getvalue(), uploaded_file.name)
    time_arr = data['time']
    t_min, t_max = float(time_arr[0]), float(time_arr[-1])
    
    onsets_final = {}
    cols_ui = st.columns(2)

    for i, (ch, label, ui_col) in enumerate(zip(['CH1', 'CH2'], labels, cols_ui)):
        with ui_col:
            st.subheader(label)
            
            # SLIDER 100% FIÁVEL (Substitui o click no gráfico)
            st.info("Deslize as barras abaixo para selecionar o trecho de contração:")
            selected_range = st.slider(
                f"Intervalo {label}", 
                min_value=t_min, max_value=t_max, 
                value=(t_min + 0.5, t_max - 0.5), # Posição inicial sugerida
                step=0.01, 
                key=f"slider_{i}"
            )
            t_start, t_end = selected_range
            
            # Recorta os dados para análise
            mask = (time_arr >= t_start) & (time_arr <= t_end)
            seg_time = time_arr[mask]
            seg_rms = data[ch][mask]
            
            # Prepara o gráfico visual
            fig = go.Figure()
            # Sinal Inteiro
            fig.add_trace(go.Scatter(x=time_arr, y=data[ch], line=dict(color='lightgray', width=1), name="Sinal Original"))
            # Trecho Selecionado (Destaque)
            fig.add_trace(go.Scatter(x=seg_time, y=seg_rms, line=dict(color='black', width=1.5), name="Área Selecionada"))
            
            # Marcação da Baseline Fixa (0-200ms)
            fig.add_vrect(x0=0, x1=0.2, fillcolor="cyan", opacity=0.3, line_width=0, annotation_text="Baseline (200ms)")
            # Marcação da Área de Análise
            fig.add_vrect(x0=t_start, x1=t_end, fillcolor="purple", opacity=0.1, line_width=1, annotation_text="Análise")
            
            fig.update_layout(
                height=300, margin=dict(l=0, r=0, t=20, b=0),
                paper_bgcolor='white', plot_bgcolor='white',
                xaxis=dict(showgrid=True, gridcolor='#eee', title="Tempo (s)"),
                yaxis=dict(showgrid=True, gridcolor='#eee'),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # REALIZA OS CÁLCULOS DO PROTOCOLO UFMG
            base_info = baselines[ch]
            idx_on = detect_onset(seg_rms, base_info['thresh'], fs)
            
            if len(seg_rms) > 0:
                v_peak = np.max(seg_rms)
                t_peak = seg_time[np.argmax(seg_rms)]
                v_area = simpson(seg_rms, dx=1/fs)
                
                # Relatório
                st.markdown(f"""
                <div class="report-box">
                    <strong>RELATÓRIO: {label.upper()}</strong><br>
                    Intervalo: {t_start:.2f}s - {t_end:.2f}s<br><br>
                    • 🟢 ONSET: {seg_time[idx_on] if idx_on is not None else "N/D"} s<br>
                    • 🔴 PICO MÁXIMO: {v_peak:.2f} µV<br>
                    • ⏱️ TEMPO DO PICO: {t_peak:.4f} s<br>
                    • 📊 ÁREA (INTEGRAL): {v_area:.4f} µV.s<br>
                    <hr>
                    <small><i>Baseline Fixa (0-200ms): Média {base_info['mean']:.2f} | Thresh {base_info['thresh']:.2f} µV</i></small>
                </div>
                """, unsafe_allow_html=True)
                
                if idx_on is not None:
                    onsets_final[i] = seg_time[idx_on]

    # SINCRONISMO FINAL
    if len(onsets_final) == 2:
        st.write("---")
        delay = abs(onsets_final[0] - onsets_final[1]) * 1000
        st.subheader(f"⏱️ Diferença de Sincronismo (Delay): **{delay:.2f} ms**")

else:
    st.info("Aguardando arquivo para iniciar.")
