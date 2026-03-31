import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt
from scipy.integrate import simpson

# 1. CONFIGURAÇÃO DE INTERFACE
st.set_page_config(page_title="Análise EMG UFMG", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: white; color: black; }
    .report-box { 
        border: 2px solid #000; padding: 15px; 
        background-color: #f8f9fa; color: black;
        font-family: 'Courier New', monospace; margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. FUNÇÕES TÉCNICAS RÍGIDAS (PROTOCOLO UFMG)
def process_emg_signal(data, fs=2000):
    nyq = 0.5 * fs
    # Filtro Butterworth 4ª ordem (6-500Hz)
    b, a = butter(4, [6/nyq, 500/nyq], btype='bandpass')
    filt = filtfilt(b, a, data)
    rect = np.abs(filt - np.mean(filt))
    # Janela RMS de 10ms
    window = int(fs * 0.01)
    rms = np.sqrt(np.convolve(rect**2, np.ones(window)/window, mode='same'))
    return rms

def get_onset(rms_seg, full_rms, fs=2000):
    # Baseline: Primeiros 400 pontos (repouso)
    baseline = full_rms[:400]
    thresh = np.mean(baseline) + (3 * np.std(baseline))
    # Busca Onset (20ms acima do threshold)
    for i in range(len(rms_seg) - 40):
        if np.all(rms_seg[i : i + 40] >= thresh):
            return i, thresh
    return None, thresh

# 3. LÓGICA DE CARREGAMENTO
uploaded_file = st.sidebar.file_uploader("Carregar arquivo (.slk ou .csv)", type=["slk", "csv"])

if uploaded_file:
    # Garante que os dados fiquem salvos na sessão para não perder a seleção
    if 'data_df' not in st.session_state or st.session_state.get('fname') != uploaded_file.name:
        content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        raw_list = []
        for line in content.splitlines():
            if line.startswith('C;'):
                p = line.split(';')
                try: raw_list.append([int(p[1][1:]), int(p[2][1:]), float(p[3].replace('"',''))])
                except: continue
        
        df_raw = pd.DataFrame(raw_list, columns=['r', 'c', 'v'])
        # Organiza colunas: Tempo (1), Canal 1 (4), Canal 2 (5)
        t = df_raw[df_raw['c'] == 1]['v'].values
        ch1 = df_raw[df_raw['c'] == 4]['v'].values
        ch2 = df_raw[df_raw['c'] == 5]['v'].values
        
        # Ajuste de tamanho caso as colunas tenham comprimentos diferentes
        min_len = min(len(t), len(ch1), len(ch2))
        st.session_state.data_df = pd.DataFrame({'time': t[:min_len], 'CH1': ch1[:min_len], 'CH2': ch2[:min_len]})
        st.session_state.fname = uploaded_file.name

    df = st.session_state.data_df
    onsets_results = {}
    cols = st.columns(2)

    for i, ch_key in enumerate(['CH1', 'CH2']):
        with cols[i]:
            st.subheader(f"Canal {i+1}")
            
            # Processa o sinal completo para visualização
            rms_total = process_emg_signal(df[ch_key].values)
            
            # Gráfico otimizado para SELEÇÃO MANUAL
            fig = go.Figure(go.Scatter(x=df['time'], y=rms_total, line=dict(color='black', width=1)))
            fig.update_layout(
                height=350, margin=dict(l=0, r=0, t=0, b=0),
                dragmode='select', selectdirection='h',
                plot_bgcolor='white'
            )
            
            # GATILHO DE SELEÇÃO
            # Usamos "box" em vez de "points" porque é muito mais leve
            sel_event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key=f"p_{ch_key}")

            # Se houver seleção manual (caixa desenhada no gráfico)
            if sel_event and "selection" in sel_event and "box" in sel_event["selection"] and len(sel_event["selection"]["box"]) > 0:
                # Pega os limites da caixa (Tempo inicial e final)
                t_start = sel_event["selection"]["box"][0]["x"][0]
                t_end = sel_event["selection"]["box"][0]["x"][1]
                
                # Filtra o trecho
                mask = (df['time'] >= t_start) & (df['time'] <= t_end)
                seg_rms = rms_total[mask]
                seg_time = df['time'][mask].values
                
                if len(seg_rms) > 0:
                    idx, thr = get_onset(seg_rms, rms_total)
                    v_peak = np.max(seg_rms)
                    v_area = simpson(seg_rms, dx=1/2000)
                    
                    st.markdown(f"""
                    <div class="report-box">
                        <strong>ANÁLISE CANAL {i+1}</strong><br>
                        Trecho: {t_start:.2f}s - {t_end:.2f}s<br><br>
                        • ONSET: {seg_time[idx] if idx is not None else "N/D"} s<br>
                        • PICO: {v_peak:.2f} µV<br>
                        • ÁREA: {v_area:.4f} µV.s<br>
                        • THRESHOLD: {thr:.4f} µV
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if idx is not None:
                        onsets_results[i] = seg_time[idx]
            else:
                st.warning("👆 Selecione um trecho no gráfico para analisar.")

    # Sincronismo entre canais
    if len(onsets_results) == 2:
        diff = abs(onsets_results[0] - onsets_results[1]) * 1000
        st.success(f"### Diferença de Sincronismo: {diff:.2f} ms")
else:
    st.info("Aguardando upload do arquivo...")
