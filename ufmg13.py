import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt
from scipy.integrate import simpson
import io

# ==========================================================
# 1. ESTILO MINIMALISTA (BRANCO E PRETO)
# ==========================================================
st.set_page_config(page_title="EMG Analysis UFMG", layout="wide")

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
        line-height: 1.5;
    }
    hr { border: 0.5px solid #000; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================================
# 2. FUNÇÕES TÉCNICAS (DO ARQUIVO UFMG.PY ORIGINAL)
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
                # Identificação de nomes na linha 4
                if row == 4 and col in [4, 5]:
                    muscle_names[col] = val
            except: continue
    df = pd.DataFrame(data).T.sort_index()
    df = df.iloc[4:, [0, 3, 4]].reset_index(drop=True).astype(float)
    df.columns = ['time', 'CH1', 'CH2']
    return df, muscle_names

def process_signal(raw_signal, fs=2000):
    # Filtro Butterworth 4ª ordem (6-500Hz)
    b, a = butter(4, [6/(fs/2), 500/(fs/2)], btype='bandpass')
    filtered = filtfilt(b, a, raw_signal)
    # Retificação e RMS de 10ms
    rectified = np.abs(filtered - np.mean(filtered))
    window = int(fs * 0.01)
    rms = np.sqrt(np.convolve(rectified**2, np.ones(window)/window, mode='same'))
    return rms

def detect_onset(signal, fs=2000):
    base_mean = np.mean(signal[:400])
    base_std = np.std(signal[:400])
    thresh = base_mean + (3 * base_std)
    for i in range(len(signal) - 40): # 20ms sustentado
        if np.all(signal[i : i + 40] >= thresh):
            return i, thresh
    return None, thresh

# ==========================================================
# 3. INTERFACE E ANÁLISE INDIVIDUALIZADA
# ==========================================================

st.title("Sistema de Análise EMG - Protocolo UFMG")
st.write("---")

uploaded_file = st.sidebar.file_uploader("Upload de arquivo (.slk ou .csv)", type=["slk", "csv"])

if uploaded_file:
    # Carregamento
    if uploaded_file.name.endswith('.slk'):
        df, names = read_slk_file(uploaded_file)
        m_labels = [names.get(4, "Canal 1"), names.get(5, "Canal 2")]
    else:
        df = pd.read_csv(uploaded_file)
        df.columns = ['time', 'CH1', 'CH2']
        m_labels = ["Músculo 1", "Músculo 2"]

    fs = 2000
    time_full = df['time'].values
    onsets_found = {}

    col1, col2 = st.columns(2)

    # LOOP PARA CADA MÚSCULO COM SUA PRÓPRIA SELEÇÃO
    for i, (col_name, label, display_col) in enumerate(zip(['CH1', 'CH2'], m_labels, [col1, col2])):
        with display_col:
            st.subheader(label)
            
            # Seleção de Intervalo Individual
            t_start, t_end = st.slider(
                f"Intervalo de Análise - {label}",
                float(time_full.min()), float(time_full.max()), 
                (float(time_full.min()), float(time_full.max())), key=f"slide_{i}"
            )

            # Recorte e Processamento
            mask = (df['time'] >= t_start) & (df['time'] <= t_end)
            df_sub = df.loc[mask].reset_index(drop=True)
            t_sub = df_sub['time'].values
            
            rms_data = process_signal(df_sub[col_name].values, fs)
            idx, thr = detect_onset(rms_data, fs)
            
            # Gráfico
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t_sub, y=rms_data, name="RMS", line=dict(color='black', width=1)))
            if idx is not None:
                onsets_found[i] = t_sub[idx]
                fig.add_vline(x=t_sub[idx], line_dash="dash", line_color="red")
            
            fig.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor='white', plot_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)

            # --- ANÁLISE COMPLETA (DADOS DO ARQUIVO ORIGINAL) ---
            peak_val = np.max(rms_data)
            peak_time = t_sub[np.argmax(rms_data)]
            mean_rms = np.mean(rms_data)
            area = simpson(rms_data, dx=1/fs)

            st.markdown(f"""
            <div class="report-box">
                <strong>DADOS DE ANÁLISE:</strong><br>
                • ONSET: {t_sub[idx] if idx is not None else "N/D"} s<br>
                • PICO MÁXIMO: {peak_val:.2f} µV<br>
                • TEMPO DO PICO: {peak_time:.4f} s<br>
                • RMS MÉDIO: {mean_rms:.2f} µV<br>
                • ÁREA (INTEGRAL): {area:.4f} µV.s<br>
                • THRESHOLD BASE: {thr:.4f} µV
            </div>
            """, unsafe_allow_html=True)

    # 4. COMPARATIVO DE SINCRONISMO (DELAY)
    if len(onsets_found) == 2:
        st.write("---")
        st.subheader("Comparativo de Sincronismo")
        delay = abs(onsets_found[0] - onsets_found[1]) * 1000
        primeiro = m_labels[0] if onsets_found[0] < onsets_found[1] else m_labels[1]
        
        st.write(f"Primeiro músculo ativado: **{primeiro}**")
        st.write(f"Diferença de ativação (Delay): **{delay:.2f} ms**")
        st.write("---")

else:
    st.info("Aguardando upload de arquivo para processamento.")