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
        border: 2px solid #000; 
        padding: 20px; 
        font-family: 'Courier New', Courier, monospace; 
        margin-top: 10px;
        background-color: #fff;
        color: #000;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================================
# 2. FUNÇÕES TÉCNICAS
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

def process_emg_core(signal, fs=2000):
    nyq = 0.5 * fs
    b, a = butter(4, [6/nyq, 500/nyq], btype='bandpass')
    filt = filtfilt(b, a, signal)
    rect = np.abs(filt - np.mean(filt))
    window = int(fs * 0.01) # 10ms
    rms = np.sqrt(np.convolve(rect**2, np.ones(window)/window, mode='same'))
    return rms

def detect_onset_core(rms_signal, fs=2000):
    if len(rms_signal) < 400: return None, 0
    base_mean = np.mean(rms_signal[:400])
    base_std = np.std(rms_signal[:400])
    thresh = base_mean + (3 * base_std)
    for i in range(len(rms_signal) - 40):
        if np.all(rms_signal[i : i + 40] >= thresh):
            return i, thresh
    return None, thresh

# ==========================================================
# 3. INTERFACE E LÓGICA DE SELEÇÃO
# ==========================================================
st.title("Análise de EMG - Protocolo UFMG")
st.write("---")

uploaded_file = st.sidebar.file_uploader("Carregar arquivo (.slk ou .csv)", type=["slk", "csv"])

if uploaded_file:
    # Garantir que os dados fiquem na memória
    if 'raw_df' not in st.session_state or st.session_state.get('last_file') != uploaded_file.name:
        if uploaded_file.name.endswith('.slk'):
            df, names = read_slk_file(uploaded_file)
            st.session_state.m_labels = [names.get(4, "Canal 1"), names.get(5, "Canal 2")]
        else:
            df = pd.read_csv(uploaded_file)
            df.columns = ['time', 'CH1', 'CH2']
            st.session_state.m_labels = ["Canal 1", "Canal 2"]
        st.session_state.raw_df = df
        st.session_state.last_file = uploaded_file.name

    df_full = st.session_state.raw_df
    m_labels = st.session_state.m_labels
    fs = 2000
    onsets_final = {}
    
    col_a, col_b = st.columns(2)

    for i, (ch, label, ui_col) in enumerate(zip(['CH1', 'CH2'], m_labels, [col_a, col_b])):
        with ui_col:
            st.subheader(label)
            
            # Sinal completo para o gráfico
            full_rms = process_emg_core(df_full[ch].values, fs)
            
            fig = go.Figure(go.Scatter(x=df_full['time'], y=full_rms, line=dict(color='black', width=1)))
            fig.update_layout(
                height=350, margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor='white', plot_bgcolor='white',
                dragmode='select', selectdirection='h'
            )
            
            # Captura o evento de seleção (IMPORTANT: on_select="rerun")
            event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key=f"plot_{i}")
            
            # Tenta obter o intervalo de tempo da seleção
            t_start, t_end = None, None
            if event and 'selection' in event:
                if 'range' in event['selection'] and 'x' in event['selection']['range']:
                    t_start, t_end = event['selection']['range']['x']
            
            # SE HOUVER SELEÇÃO, REALIZA A ANÁLISE
            if t_start is not None and t_end is not None:
                mask = (df_full['time'] >= t_start) & (df_full['time'] <= t_end)
                df_seg = df_full.loc[mask].reset_index(drop=True)
                
                if not df_seg.empty:
                    seg_rms = process_emg_core(df_seg[ch].values, fs)
                    idx_on, thr = detect_onset_core(seg_rms, fs)
                    
                    # Métricas
                    v_peak = np.max(seg_rms)
                    t_peak = df_seg['time'].iloc[np.argmax(seg_rms)]
                    v_area = simpson(seg_rms, dx=1/fs)
                    
                    st.markdown(f"""
                    <div class="report-box">
                        <strong>RELATÓRIO: {label.upper()}</strong><br>
                        Intervalo: {t_start:.2f}s - {t_end:.2f}s<br><br>
                        • ONSET: {df_seg['time'].iloc[idx_on] if idx_on is not None else "N/D"} s<br>
                        • PICO: {v_peak:.2f} µV<br>
                        • TEMPO DO PICO: {t_peak:.4f} s<br>
                        • RMS MÉDIO: {np.mean(seg_rms):.2f} µV<br>
                        • ÁREA: {v_area:.4f} µV.s<br>
                        • THRESHOLD: {thr:.4f} µV
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if idx_on is not None:
                        onsets_final[i] = df_seg['time'].iloc[idx_on]
            else:
                st.info("👆 Selecione um intervalo no gráfico acima para analisar.")

    # Sincronismo
    if len(onsets_final) == 2:
        st.write("---")
        delay = abs(onsets_final[0] - onsets_final[1]) * 1000
        st.subheader(f"Diferença de Sincronismo: {delay:.2f} ms")

else:
    st.info("Aguardando ficheiro para começar.")
