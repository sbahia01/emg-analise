import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt
from scipy.integrate import simpson
import io

# ==========================================================
# 1. CONFIGURAÇÕES GERAIS E ESTILO (PADRÃO UFMG)
# ==========================================================
st.set_page_config(page_title="Análise EMG UFMG - Sistema Profissional", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #ffffff; color: #000000; }
    .report-box { 
        border: 2px solid #000000; 
        padding: 20px; 
        background-color: #f9f9f9; 
        color: #000000;
        font-family: 'Courier New', Courier, monospace;
        margin-top: 15px;
        box-shadow: 4px 4px 0px #cccccc;
    }
    .status-msg { font-weight: bold; color: #2e7d32; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================================
# 2. FUNÇÕES TÉCNICAS ORIGINAIS (FILTROS E CÁLCULOS)
# ==========================================================

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_emg_filter(data, fs=2000):
    """Aplica o filtro Butterworth de 4ª ordem (6-500Hz) e retificação."""
    if len(data) < 15: # Proteção contra dados curtos
        return data
    b, a = butter_bandpass(6, 500, fs, order=4)
    # Filtro e remoção de Offset (Média)
    y = filtfilt(b, a, data)
    rectified = np.abs(y - np.mean(y))
    return rectified

def calculate_rms_signal(rectified_data, fs=2000, window_ms=10):
    """Calcula o sinal RMS com janela de 10ms conforme protocolo UFMG."""
    window_size = int(fs * (window_ms / 1000))
    if window_size < 1: window_size = 1
    # Convolução para média móvel quadrática
    rms = np.sqrt(np.convolve(rectified_data**2, np.ones(window_size)/window_size, mode='same'))
    return rms

def detect_onset(rms_segment, full_rms, fs=2000):
    """Detecta Onset usando baseline fixa (400 pontos iniciais)."""
    # Baseline: Primeiros 200ms (400 pontos a 2000Hz)
    baseline = full_rms[:400]
    thresh = np.mean(baseline) + (3 * np.std(baseline))
    
    # Busca 20ms consecutivos acima do threshold
    check_samples = int(0.020 * fs) 
    for i in range(len(rms_segment) - check_samples):
        if np.all(rms_segment[i : i + check_samples] >= thresh):
            return i, thresh
    return None, thresh

# ==========================================================
# 3. LEITURA DE ARQUIVOS .SLK (ROBUSTEZ TOTAL)
# ==========================================================

def parse_slk_file(uploaded_file):
    """Lê arquivos SYLK (.slk) extraindo tempo e canais 4 e 5."""
    try:
        content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        lines = content.splitlines()
        
        data_map = {}
        ch_names = {4: "Canal 1", 5: "Canal 2"}
        
        for line in lines:
            if line.startswith('C;'):
                parts = line.split(';')
                try:
                    # C;Yrow;Xcol;Kvalue
                    r = int(parts[1][1:])
                    c = int(parts[2][1:])
                    v_part = parts[3]
                    
                    if v_part.startswith('K'):
                        val = float(v_part[1:].replace('"', ''))
                    else:
                        val = float(v_part.replace('"', ''))
                    
                    if r not in data_map: data_map[r] = {}
                    data_map[r][c] = val
                    
                    # Captura nomes de músculos na linha 4
                    if r == 4 and c in [4, 5] and v_part.startswith('K'):
                        ch_names[c] = v_part[1:].replace('"', '')
                except: continue
        
        # Organiza os dados em DataFrame
        df_rows = pd.DataFrame.from_dict(data_map, orient='index').sort_index()
        df_data = df_rows.loc[df_rows.index > 5].copy() # Pula cabeçalhos
        
        final_df = pd.DataFrame({
            'time': df_data[1].values,
            'CH1': df_data[4].values,
            'CH2': df_data[5].values
        }).dropna()
        
        return final_df, [ch_names[4], ch_names[5]]
    except Exception as e:
        st.error(f"Erro ao processar arquivo .slk: {e}")
        return None, None

# ==========================================================
# 4. INTERFACE E SELEÇÃO MANUAL VIA PLOTLY
# ==========================================================

st.title("Análise de EMG - Protocolo UFMG Profissional")
st.sidebar.header("Configurações")

file = st.sidebar.file_uploader("Carregar arquivo (.slk ou .csv)", type=["slk", "csv"])

if file:
    # Cache para não reprocessar o arquivo a cada clique no gráfico
    if 'data' not in st.session_state or st.session_state.get('filename') != file.name:
        df, labels = parse_slk_file(file)
        if df is not None:
            st.session_state.data = df
            st.session_state.labels = labels
            st.session_state.filename = file.name
        else:
            st.stop()

    df = st.session_state.data
    labels = st.session_state.labels
    fs = 2000
    onsets_final = {}

    col1, col2 = st.columns(2)

    for i, (ch_key, label) in enumerate(zip(['CH1', 'CH2'], labels)):
        with (col1 if i == 0 else col2):
            st.subheader(label)
            
            # Processamento rigoroso
            rectified = apply_emg_filter(df[ch_key].values, fs)
            rms_full = calculate_rms_signal(rectified, fs)
            
            # Gráfico interativo configurado para SELEÇÃO MANUAL
            fig = go.Figure(go.Scatter(x=df['time'], y=rms_full, line=dict(color='black', width=1)))
            fig.update_layout(
                height=400, margin=dict(l=10, r=10, t=10, b=10),
                dragmode='select', selectdirection='h', # Habilita seleção horizontal
                plot_bgcolor='white',
                xaxis=dict(showgrid=True, gridcolor='#f0f0f0', title="Tempo (s)"),
                yaxis=dict(showgrid=True, gridcolor='#f0f0f0', title="µV")
            )
            
            # Gatilho de seleção manual
            # Importante: on_select="rerun" faz o Streamlit capturar a caixa desenhada
            sel_event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key=f"sel_{ch_key}")

            # ANÁLISE DOS DADOS SELECIONADOS
            if sel_event and "selection" in sel_event and "box" in sel_event["selection"] and len(sel_event["selection"]["box"]) > 0:
                t_in = sel_event["selection"]["box"][0]["x"][0]
                t_out = sel_event["selection"]["box"][0]["x"][1]
                
                # Recorte temporal
                mask = (df['time'] >= t_in) & (df['time'] <= t_out)
                seg_time = df['time'][mask].values
                seg_rms = rms_full[mask]
                
                if len(seg_rms) > 40:
                    idx, thr = detect_onset(seg_rms, rms_full, fs)
                    v_max = np.max(seg_rms)
                    t_max = seg_time[np.argmax(seg_rms)]
                    area = simpson(seg_rms, dx=1/fs)
                    
                    st.markdown(f"""
                    <div class="report-box">
                        <strong>RELATÓRIO TÉCNICO: {label.upper()}</strong><br><br>
                        • Intervalo Selecionado: {t_in:.2f}s - {t_out:.2f}s<br>
                        • 🟢 <b>ONSET:</b> {seg_time[idx] if idx is not None else "N/D"} s<br>
                        • 🔴 <b>PICO MÁXIMO:</b> {v_max:.2f} µV<br>
                        • ⏱️ <b>TEMPO DO PICO:</b> {t_max:.4f} s<br>
                        • 📊 <b>ÁREA (INTEGRAL):</b> {area:.4f} µV.s<br>
                        <hr>
                        <small>Threshold de detecção: {thr:.4f} µV</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if idx is not None:
                        onsets_final[i] = seg_time[idx]
            else:
                st.info(f"🖱️ Clique e arraste no gráfico do **{label}** para realizar a análise.")

    # CÁLCULO DE DELAY (SINCRONISMO)
    if len(onsets_final) == 2:
        st.write("---")
        delay = abs(onsets_final[0] - onsets_final[1]) * 1000
        st.success(f"### ⏱️ Diferença de Sincronismo (Delay): **{delay:.2f} ms**")

else:
    st.info("Aguardando upload do arquivo para iniciar a análise.")
