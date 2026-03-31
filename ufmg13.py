import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt
from scipy.integrate import simpson

# ==========================================================
# 1. CONFIGURAÇÃO VISUAL (UFMG PADRÃO)
# ==========================================================
st.set_page_config(page_title="Análise EMG UFMG", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: white !important; color: black !important; }
    .report-box { 
        border: 2px solid #000; 
        padding: 15px; 
        font-family: 'Courier New', Courier, monospace; 
        background-color: #fff;
        color: #000;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================================
# 2. PROCESSAMENTO DE SINAL
# ==========================================================

def process_emg(signal, fs=2000):
    nyq = 0.5 * fs
    b, a = butter(4, [6/nyq, 500/nyq], btype='bandpass')
    filt = filtfilt(b, a, signal)
    rect = np.abs(filt - np.mean(filt))
    window = int(fs * 0.01) # RMS 10ms
    return np.sqrt(np.convolve(rect**2, np.ones(window)/window, mode='same'))

def detect_onset(rms_seg, full_rms, fs=2000):
    # Baseline: Primeiros 400 pontos do sinal COMPLETO (Protocolo UFMG)
    base_mean = np.mean(full_rms[:400])
    base_std = np.std(full_rms[:400])
    thresh = base_mean + (3 * base_std)
    
    for i in range(len(rms_seg) - 40):
        if np.all(rms_seg[i : i + 40] >= thresh):
            return i, thresh
    return None, thresh

# ==========================================================
# 3. INTERFACE E SELEÇÃO MANUAL
# ==========================================================
st.title("Análise de EMG - Seleção Manual")
st.info("🖱️ **Como usar:** Clique e arraste o mouse sobre a área da contração em cada gráfico para ver a análise técnica.")

uploaded_file = st.sidebar.file_uploader("Carregar arquivo", type=["slk", "csv"])

if uploaded_file:
    # Carregamento de dados (Persistente na sessão)
    if 'df' not in st.session_state or st.session_state.filename != uploaded_file.name:
        content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        data = {}
        names = {}
        for line in content.splitlines():
            if line.startswith('C;'):
                parts = line.strip().split(';')
                try:
                    r, c = int(parts[1][1:]), int(parts[2][1:])
                    v = parts[3][1:].strip('"') if parts[3].startswith('K') else parts[3]
                    if r not in data: data[r] = {}
                    data[r][c] = v
                    if r == 4 and c in [4, 5]: names[c] = v
                except: continue
        df = pd.DataFrame(data).T.sort_index().iloc[4:, [0, 3, 4]].astype(float)
        df.columns = ['time', 'CH1', 'CH2']
        st.session_state.df = df
        st.session_state.labels = [names.get(4, "CH1"), names.get(5, "CH2")]
        st.session_state.filename = uploaded_file.name

    df = st.session_state.df
    onsets = {}
    cols = st.columns(2)

    for i, (ch, label) in enumerate(zip(['CH1', 'CH2'], st.session_state.labels)):
        with cols[i]:
            st.subheader(label)
            
            # Cálculo do sinal para o gráfico
            rms_full = process_emg(df[ch].values)
            
            fig = go.Figure(go.Scatter(x=df['time'], y=rms_full, line=dict(color='black', width=1)))
            fig.update_layout(
                height=350, margin=dict(l=10, r=10, t=10, b=10),
                dragmode='select', selectdirection='h', # Habilita seleção manual horizontal
                paper_bgcolor='white', plot_bgcolor='white',
                xaxis=dict(showgrid=True, gridcolor='#eee'),
                yaxis=dict(showgrid=True, gridcolor='#eee')
            )
            
            # Captura o evento de seleção manual do mouse
            selected_data = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key=f"manual_{i}")

            # LÓGICA DE ANÁLISE PÓS-SELEÇÃO
            if selected_data and "selection" in selected_data and len(selected_data["selection"]["points"]) > 0:
                # Obtém o intervalo exato selecionado pelo mouse
                x_values = [p["x"] for p in selected_data["selection"]["points"]]
                t_start, t_end = min(x_values), max(x_values)
                
                # Filtra os dados brutos para este trecho
                mask = (df['time'] >= t_start) & (df['time'] <= t_end)
                df_seg = df.loc[mask]
                rms_seg = process_emg(df_seg[ch].values)
                
                # Cálculos técnicos
                idx, thr = detect_onset(rms_seg, rms_full)
                v_peak = np.max(rms_seg)
                t_peak = df_seg['time'].iloc[np.argmax(rms_seg)]
                v_area = simpson(rms_seg, dx=1/2000)
                
                st.markdown(f"""
                <div class="report-box">
                    <strong>ANÁLISE MANUAL: {label}</strong><br>
                    Intervalo: {t_start:.2f}s a {t_end:.2f}s<br><br>
                    • ONSET: {df_seg['time'].iloc[idx] if idx is not None else "N/D"} s<br>
                    • PICO: {v_peak:.2f} µV<br>
                    • ÁREA: {v_area:.4f} µV.s<br>
                    • THRESHOLD: {thr:.4f} µV
                </div>
                """, unsafe_allow_html=True)
                
                if idx is not None:
                    onsets[i] = df_seg['time'].iloc[idx]
            else:
                st.caption("Aguardando seleção manual no gráfico...")

    # Delay de Sincronismo
    if len(onsets) == 2:
        st.write("---")
        delay = abs(onsets[0] - onsets[1]) * 1000
        st.subheader(f"Diferença de Sincronismo: {delay:.2f} ms")

else:
    st.info("Por favor, carregue o arquivo para começar.")
