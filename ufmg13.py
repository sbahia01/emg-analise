import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt
from scipy.integrate import simpson

# ==========================================================
# CONFIGURAÇÃO VISUAL (MINIMALISTA - BRANCO E PRETO)
# ==========================================================
st.set_page_config(page_title="EMG UFMG Analysis", layout="wide")

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
# FUNÇÕES TÉCNICAS (CONSOLIDADO UFMG.PY)
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

def process_emg(signal, fs=2000):
    b, a = butter(4, [6/(fs/2), 500/(fs/2)], btype='bandpass')
    filt = filtfilt(b, a, signal)
    rect = np.abs(filt - np.mean(filt))
    window = int(fs * 0.01) # 10ms
    return np.sqrt(np.convolve(rect**2, np.ones(window)/window, mode='same'))

def detect_onset(signal, time_array, fs=2000):
    if len(signal) < 400: return None, 0
    base_mean = np.mean(signal[:400])
    base_std = np.std(signal[:400])
    thresh = base_mean + (3 * base_std)
    for i in range(len(signal) - 40):
        if np.all(signal[i : i + 40] >= thresh):
            return i, thresh
    return None, thresh

# ==========================================================
# INTERFACE DO USUÁRIO
# ==========================================================
st.title("Análise EMG - Seleção Manual por Gráfico")
st.info("Instrução: Use o rato para selecionar (clicar e arrastar) a área de interesse diretamente no gráfico.")

uploaded_file = st.sidebar.file_uploader("Arquivo (.slk ou .csv)", type=["slk", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith('.slk'):
        df, names = read_slk_file(uploaded_file)
        m_labels = [names.get(4, "Músculo 1"), names.get(5, "Músculo 2")]
    else:
        df = pd.read_csv(uploaded_file)
        df.columns = ['time', 'CH1', 'CH2']
        m_labels = ["Canal 1", "Canal 2"]

    fs = 2000
    onsets_log = {}
    cols = st.columns(2)

    for i, (ch, label, col_ui) in enumerate(zip(['CH1', 'CH2'], m_labels, cols)):
        with col_ui:
            st.subheader(label)
            
            # --- GRÁFICO INTERATIVO PARA SELEÇÃO ---
            # Processamos o sinal inteiro para exibição inicial
            full_rms = process_emg(df[ch].values, fs)
            
            fig = go.Figure(go.Scatter(x=df['time'], y=full_rms, line=dict(color='black', width=1)))
            fig.update_layout(
                height=350, 
                margin=dict(l=0,r=0,t=10,b=0), 
                paper_bgcolor='white', 
                plot_bgcolor='white',
                dragmode='select', # Habilita a seleção por caixa por padrão
                selectdirection='h' # Seleção apenas horizontal (tempo)
            )
            
            # Captura a seleção feita pelo usuário no gráfico
            selected_data = st.plotly_chart(fig, use_container_width=True, on_select="rerun")
            
            # --- LÓGICA DE RECORTE E ANÁLISE ---
            # Se o usuário selecionou uma área, filtramos. Se não, usamos o sinal todo.
            if selected_data and "selection" in selected_data and len(selected_data["selection"]["points"]) > 0:
                # Pegamos o tempo mínimo e máximo da seleção
                x_values = [p["x"] for p in selected_data["selection"]["points"]]
                t_min, t_max = min(x_values), max(x_values)
                
                df_sub = df[(df['time'] >= t_min) & (df['time'] <= t_max)].reset_index(drop=True)
                st.caption(f"Intervalo selecionado: {t_min:.3f}s a {t_max:.3f}s")
            else:
                df_sub = df
                st.caption("Exibindo sinal total. Selecione uma área no gráfico para analisar.")

            t_sub = df_sub['time'].values
            rms_sub = process_emg(df_sub[ch].values, fs)
            idx, thr = detect_onset(rms_sub, t_sub, fs)
            
            # Cálculos
            peak = np.max(rms_sub)
            area = simpson(rms_sub, dx=1/fs)
            
            st.markdown(f"""
            <div class="report-box">
                <strong>RESULTADOS {label.upper()}:</strong><br>
                Onset: {t_sub[idx] if idx is not None else "N/D"} s<br>
                Pico Máx: {peak:.2f} µV<br>
                RMS Médio: {np.mean(rms_sub):.2f} µV<br>
                Área: {area:.4f} µV.s<br>
                Threshold: {thr:.4f} µV
            </div>
            """, unsafe_allow_html=True)
            
            if idx is not None: onsets_log[i] = t_sub[idx]

    # SINCRONISMO
    if len(onsets_log) == 2:
        st.write("---")
        delay = abs(onsets_log[0] - onsets_log[1]) * 1000
        st.write(f"### Diferença de Sincronismo (Delay): **{delay:.2f} ms**")
else:
    st.info("Aguardando upload de dados...")
