import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt
from scipy.integrate import simpson

# 1. SETUP DE PÁGINA
st.set_page_config(page_title="Análise EMG UFMG", layout="wide")

# Estilo para garantir visibilidade dos resultados
st.markdown("""
    <style>
    .report-card { 
        background-color: #f8f9fa; 
        border-left: 5px solid #28a745; 
        padding: 20px; 
        margin: 10px 0;
        color: black;
        font-family: monospace;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. FUNÇÕES DE PROCESSAMENTO (Protocolo UFMG Completo)
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def process_signal(data, fs=2000):
    b, a = butter_bandpass(6, 500, fs, order=4)
    # Filtragem e retificação
    filt = filtfilt(b, a, data)
    rect = np.abs(filt - np.mean(filt))
    # RMS - Janela móvel de 10ms (20 pontos)
    window = int(fs * 0.01)
    rms = np.sqrt(np.convolve(rect**2, np.ones(window)/window, mode='same'))
    return rms

def get_onset(rms_segment, full_rms, fs=2000):
    # Baseline baseada nos primeiros 400 pontos do sinal bruto (conforme UFMG)
    baseline = full_rms[:400]
    thresh = np.mean(baseline) + (3 * np.std(baseline))
    # Busca o Onset no segmento selecionado
    for i in range(len(rms_segment) - 40):
        if np.all(rms_segment[i:i+40] >= thresh):
            return i, thresh
    return None, thresh

# 3. CARREGAMENTO E INTERFACE
uploaded_file = st.sidebar.file_uploader("Selecione o arquivo (.slk ou .csv)", type=["slk", "csv"])

if uploaded_file:
    # Lógica de leitura (Simplificada para garantir execução)
    if uploaded_file.name.endswith('.slk'):
        content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        raw_data = []
        for line in content.splitlines():
            if line.startswith('C;'):
                p = line.split(';')
                try: raw_data.append([int(p[1][1:]), int(p[2][1:]), p[3].strip('"')])
                except: continue
        # Pivotar e organizar (Assumindo colunas 4 e 5 para músculos)
        df_temp = pd.DataFrame(raw_data, columns=['row', 'col', 'val'])
        df_temp['val'] = pd.to_numeric(df_temp['val'], errors='coerce')
        df = df_temp.pivot(index='row', columns='col', values='val').iloc[4:, [0, 3, 4]]
        df.columns = ['time', 'CH1', 'CH2']
    else:
        df = pd.read_csv(uploaded_file).iloc[3:].astype(float)
        df.columns = ['time', 'CH1', 'CH2']

    fs = 2000
    st.session_state.onsets = {}

    cols = st.columns(2)
    for i, ch in enumerate(['CH1', 'CH2']):
        with cols[i]:
            st.subheader(f"Canal {i+1}")
            rms_full = process_signal(df[ch].values, fs)
            
            # CRIANDO O GRÁFICO COM RETORNO DE SELEÇÃO
            fig = go.Figure(go.Scatter(x=df['time'], y=rms_full, line=dict(color='black', width=1)))
            fig.update_layout(
                height=350, dragmode='select', selectdirection='h',
                margin=dict(l=0, r=0, t=0, b=0), plot_bgcolor='white'
            )
            
            # Aqui é onde o Streamlit "escuta" o gráfico
            # O parâmetro selection_mode='x' força a seleção horizontal
            selected = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key=f"plot_{ch}")

            # PROCESSANDO A SELEÇÃO
            if selected and "selection" in selected and len(selected["selection"]["points"]) > 0:
                points = selected["selection"]["points"]
                t_start = min(p["x"] for p in points)
                t_end = max(p["x"] for p in points)
                
                # Recorte do sinal
                mask = (df['time'] >= t_start) & (df['time'] <= t_end)
                seg_time = df['time'][mask].values
                seg_rms = rms_full[mask]
                
                # Análise
                idx_on, thr = get_onset(seg_rms, rms_full, fs)
                v_peak = np.max(seg_rms)
                v_area = simpson(seg_rms, dx=1/fs)
                
                st.markdown(f"""
                <div class="report-card">
                    <strong>RESULTADOS CANAL {i+1}</strong><br>
                    • Onset: {seg_time[idx_on] if idx_on is not None else "Não detectado"} s<br>
                    • Pico Máximo: {v_peak:.2f} µV<br>
                    • Área (Integral): {v_area:.4f} µV.s<br>
                    • Threshold: {thr:.4f} µV
                </div>
                """, unsafe_allow_html=True)
                
                if idx_on is not None:
                    st.session_state.onsets[i] = seg_time[idx_on]
            else:
                st.warning("⚠️ Arraste o mouse sobre o gráfico para analisar.")

    # Sincronismo
    if len(st.session_state.onsets) == 2:
        delay = abs(st.session_state.onsets[0] - st.session_state.onsets[1]) * 1000
        st.success(f"### Sincronismo: {delay:.2f} ms")

else:
    st.info("Aguardando upload...")
