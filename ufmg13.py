import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt
from scipy.integrate import simpson

# ==========================================================
# 1. IDENTIDADE VISUAL HUB ACADEMICA (CORES REAIS)
# ==========================================================
HUB_NAVY = "#001a33"   # Azul Marinho Profundo
HUB_BLUE = "#1a73e8"   # Azul Vibrante para Destaques
HUB_LIGHT = "#f8f9fa"  # Cinza Muito Claro para Fundos
WHITE = "#ffffff"

st.set_page_config(page_title="EMGExpert | Hub Academica", layout="wide")

# ==========================================================
# 2. DICIONÁRIO DE TRADUÇÕES COMPLETO
# ==========================================================
LANGS = {
    "PORTUGUÊS (BRASILEIRO)": {
        "title": "EMGExpert — Ciência Encontra a Prática",
        "upload": "Carregar arquivo (.slk ou .csv)",
        "info": "🖱️ Arraste no gráfico para analisar a contração.",
        "rep": "RELATÓRIO TÉCNICO",
        "ons": "ONSET (INÍCIO)",
        "peak": "PICO MÁXIMO",
        "area": "ÁREA (INTEGRAL)",
        "sync": "Diferença de Sincronismo (Delay)",
        "wait": "Aguardando upload do arquivo...",
        "thresh": "Threshold de detecção"
    },
    "ENGLISH": {
        "title": "EMGExpert — Science Meets Practice",
        "upload": "Upload file (.slk or .csv)",
        "info": "🖱️ Drag on the chart to analyze contraction.",
        "rep": "TECHNICAL REPORT",
        "ons": "ONSET",
        "peak": "PEAK AMPLITUDE",
        "area": "AREA (INTEGRAL)",
        "sync": "Synchronization Delay",
        "wait": "Waiting for file upload...",
        "thresh": "Detection Threshold"
    },
    "ESPAÑOL": {
        "title": "EMGExpert — La Ciencia se Encuentra con la Práctica",
        "upload": "Cargar archivo (.slk o .csv)",
        "info": "🖱️ Arrastre en el gráfico para analizar.",
        "rep": "INFORME TÉCNICO",
        "ons": "COMIENZO (ONSET)",
        "peak": "PICO MÁXIMO",
        "area": "ÁREA (INTEGRAL)",
        "sync": "Diferencia de Sincronismo",
        "wait": "Esperando carga de archivo...",
        "thresh": "Umbral de detección"
    },
    "CHINESE (SIMPLIFIED)": {
        "title": "EMGExpert — 科学与实践的结合",
        "upload": "上传文件 (.slk 或 .csv)",
        "info": "🖱️ 在图表上拖动以进行分析。",
        "rep": "技术报告",
        "ons": "起始点 (Onset)",
        "peak": "最大峰值",
        "area": "面积 (积分)",
        "sync": "同步延迟",
        "wait": "等待文件上传...",
        "thresh": "检测阈值"
    }
}

# ==========================================================
# 3. ESTILIZAÇÃO CSS (INTERFACE DO SITE)
# ==========================================================
st.markdown(f"""
    <style>
    /* Fundo Principal */
    .stApp {{ background-color: {WHITE}; }}
    
    /* Barra Lateral */
    [data-testid="stSidebar"] {{
        background-color: {HUB_NAVY};
        color: {WHITE};
    }}
    [data-testid="stSidebar"] .stMarkdown p {{ color: {WHITE}; }}
    
    /* Títulos */
    h1, h2, h3 {{ 
        color: {HUB_NAVY} !important; 
        font-family: 'Inter', sans-serif;
        font-weight: 800 !important;
    }}

    /* Cartões de Relatório Estilo Hub Academica */
    .report-card {{
        border-top: 4px solid {HUB_BLUE};
        background-color: {HUB_LIGHT};
        padding: 20px;
        border-radius: 0px 0px 8px 8px;
        color: {HUB_NAVY};
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }}
    
    /* Botão de Idioma */
    .stSelectbox label {{ color: {HUB_NAVY} !important; font-weight: bold; }}
    </style>
    """, unsafe_allow_html=True)

# ==========================================================
# 4. MOTOR TÉCNICO (ROBUSTEZ TOTAL)
# ==========================================================

def butter_bandpass_filter(data, lowcut=6, highcut=500, fs=2000, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    if len(data) <= (max(len(a), len(b)) * 3): return data
    return filtfilt(b, a, data)

def calculate_rms(data, fs=2000):
    rectified = np.abs(data - np.mean(data))
    window = int(fs * 0.01) # 10ms
    return np.sqrt(np.convolve(rectified**2, np.ones(window)/window, mode='same'))

def get_onset_stats(rms_seg, full_rms, fs=2000):
    baseline = full_rms[:400]
    thresh = np.mean(baseline) + (3 * np.std(baseline))
    check_samples = int(0.02 * fs)
    for i in range(len(rms_seg) - check_samples):
        if np.all(rms_seg[i : i + check_samples] >= thresh):
            return i, thresh
    return None, thresh

def parse_sylk(file):
    try:
        content = file.getvalue().decode("utf-8", errors="ignore")
        data_map = {}
        ch_names = {4: "CH 1", 5: "CH 2"}
        for line in content.splitlines():
            if line.startswith('C;'):
                p = line.split(';')
                try:
                    r = int(p[1][1:])
                    c = int(p[2][1:])
                    val_raw = p[3]
                    val = float(val_raw[1:].replace('"','')) if val_raw.startswith('K') else float(val_raw.replace('"',''))
                    if r not in data_map: data_map[r] = {}
                    data_map[r][c] = val
                    if r == 4 and c in [4, 5]: ch_names[c] = val_raw[1:].replace('"','')
                except: continue
        
        df_raw = pd.DataFrame.from_dict(data_map, orient='index').sort_index()
        # Coluna 1: Tempo | Coluna 4: Canal 1 | Coluna 5: Canal 2
        df = pd.DataFrame({
            'time': df_raw[1].values,
            'CH1': df_raw[4].values,
            'CH2': df_raw[5].values
        }).dropna().iloc[5:] # Pula cabeçalhos
        return df, [ch_names[4], ch_names[5]]
    except Exception as e:
        st.error(f"Erro no arquivo: {e}")
        return None, None

# ==========================================================
# 5. CONSTRUÇÃO DA INTERFACE
# ==========================================================

# Seletor de Idioma no Topo Direito
header_col1, header_col2 = st.columns([4, 1])
with header_col2:
    sel_lang = st.selectbox("🌐 Language", list(LANGS.keys()))
    tr = LANGS[sel_lang]
with header_col1:
    st.title(tr["title"])

uploaded_file = st.sidebar.file_uploader(tr["upload"], type=["slk", "csv"])

if uploaded_file:
    df_emg, labels = parse_sylk(uploaded_file)
    
    if df_emg is not None:
        fs = 2000
        onsets_results = {}
        ui_cols = st.columns(2)

        for i, (ch_col, name) in enumerate(zip(['CH1', 'CH2'], labels)):
            with ui_cols[i]:
                st.subheader(name)
                
                # Processamento
                filt = butter_bandpass_filter(df_emg[ch_col].values)
                rms_signal = calculate_rms(filt)
                
                # Gráfico com cores da Hub Academica
                fig = go.Figure(go.Scatter(
                    x=df_emg['time'], y=rms_signal, 
                    line=dict(color=HUB_NAVY, width=1.3),
                    hoverinfo='skip'
                ))
                
                fig.update_layout(
                    height=400, margin=dict(l=10, r=10, t=10, b=10),
                    dragmode='select', selectdirection='h',
                    plot_bgcolor=WHITE,
                    # Cor da seleção (Azul Hub)
                    newshape=dict(line=dict(color=HUB_BLUE, width=2), fillcolor=HUB_BLUE, opacity=0.2),
                    xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
                    yaxis=dict(showgrid=True, gridcolor='#f0f0f0')
                )
                
                # O parâmetro key inclui o idioma para forçar reset ao trocar de língua
                sel = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key=f"p_{ch_col}_{sel_lang}")

                # Análise da Seleção
                if sel and "selection" in sel and "box" in sel["selection"] and len(sel["selection"]["box"]) > 0:
                    t1 = sel["selection"]["box"][0]["x"][0]
                    t2 = sel["selection"]["box"][0]["x"][1]
                    
                    mask = (df_emg['time'] >= t1) & (df_emg['time'] <= t2)
                    st_time = df_emg['time'][mask].values
                    st_rms = rms_signal[mask]
                    
                    if len(st_rms) > 10:
                        idx, thr = get_onset_stats(st_rms, rms_signal)
                        v_max = np.max(st_rms)
                        v_area = simpson(st_rms, dx=1/fs)
                        
                        st.markdown(f"""
                        <div class="report-card">
                            <b style="font-size: 1.2em;">{tr['rep']}</b><br><br>
                            • <b>{tr['ons']}:</b> {st_time[idx] if idx is not None else "N/D"} s<br>
                            • <b>{tr['peak']}:</b> {v_max:.2f} µV<br>
                            • <b>{tr['area']}:</b> {v_area:.4f} µV.s<br>
                            <hr style="border: 0.5px solid #ccc">
                            <small>{tr['thresh']}: {thr:.4f} µV</small>
                        </div>
                        """, unsafe_allow_html=True)
                        if idx is not None: onsets_results[i] = st_time[idx]
                else:
                    st.info(tr["info"])

        # Delay Sincronismo
        if len(onsets_results) == 2:
            diff = abs(onsets_results[0] - onsets_results[1]) * 1000
            st.success(f"### ⏱️ {tr['sync']}: **{diff:.2f} ms**")
    else:
        st.error("Erro ao processar os dados do arquivo.")
else:
    st.info(tr["wait"])
