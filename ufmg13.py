import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt
from scipy.integrate import simpson

# ==========================================================
# 1. IDENTIDADE VISUAL HUB ACADEMICA (CORES SOLICITADAS)
# ==========================================================
HUB_NAVY = "#001a33"   # Azul Marinho Profundo
HUB_BLUE = "#1a73e8"   # Azul Vibrante
# AZUL MAIS CLARO SOLICITADO (Fundo da área de gráficos/app)
HUB_LIGHT_BG = "#eef4ff" 
WHITE = "#ffffff"

st.set_page_config(page_title="EMGExpert | Hub Academica", layout="wide")

# ==========================================================
# 2. DICIONÁRIO DE TRADUÇÕES (INTEGRIDADE TOTAL)
# ==========================================================
LANGS = {
    "PORTUGUÊS (BRASILEIRO)": {
        "title": "EMGExpert — Ciência Encontra a Prática",
        "upload_label": "Selecione o arquivo (.slk ou .csv)",
        "info": "🖱️ Arraste no gráfico para analisar a contração.",
        "rep": "RELATÓRIO TÉCNICO DETALHADO",
        "interval": "⏱️ JANELA DE TEMPO",
        "duration": "⏳ DURAÇÃO DA SELEÇÃO",
        "ons": "🟢 ONSET (INÍCIO)",
        "peak": "📈 PICO MÁXIMO",
        "mean_rms": "🌊 RMS MÉDIO",
        "area": "📊 ÁREA (INTEGRAL)",
        "sync": "Diferença de Sincronismo (Delay)",
        "wait": "Aguardando upload do arquivo...",
        "thresh": "Threshold de detecção aplicado"
    },
    "ENGLISH": {
        "title": "EMGExpert — Science Meets Practice",
        "upload_label": "Choose file (.slk or .csv)",
        "info": "🖱️ Drag on the chart to analyze.",
        "rep": "TECHNICAL REPORT",
        "interval": "⏱️ TIME WINDOW",
        "duration": "⏳ SELECTION DURATION",
        "ons": "🟢 ONSET",
        "peak": "📈 PEAK AMPLITUDE",
        "mean_rms": "🌊 MEAN RMS",
        "area": "📊 AREA (INTEGRAL)",
        "sync": "Delay",
        "wait": "Waiting...",
        "thresh": "Threshold"
    },
    "ESPAÑOL": {
        "title": "EMGExpert — Ciencia y Práctica",
        "upload_label": "Cargar archivo (.slk o .csv)",
        "info": "🖱️ Arrastre para analizar.",
        "rep": "INFORME TÉCNICO",
        "ons": "🟢 COMIENZO (ONSET)",
        "peak": "📈 PICO MÁXIMO",
        "mean_rms": "🌊 RMS MEDIO",
        "area": "📊 ÁREA (INTEGRAL)",
        "sync": "Delay",
        "wait": "Esperando...",
        "thresh": "Umbral"
    },
    "CHINESE (SIMPLIFIED)": {
        "title": "EMGExpert — 科学与实践",
        "upload_label": "上传文件 (.slk 或 .csv)",
        "info": "🖱️ 拖动进行分析",
        "rep": "技术报告",
        "ons": "起始点",
        "peak": "最大峰值",
        "area": "面积",
        "sync": "同步延迟",
        "wait": "等待中...",
        "thresh": "阈值"
    }
}

# ==========================================================
# 3. CSS CUSTOMIZADO (SEM CORTES + NOVAS CORES)
# ==========================================================
st.markdown(f"""
    <style>
    /* FUNDO DA PÁGINA (AZUL MAIS CLARO SOLICITADO) */
    .stApp {{
        background-color: {HUB_LIGHT_BG} !important;
    }}
    
    /* BARRA LATERAL */
    [data-testid="stSidebar"] {{
        background-color: {HUB_NAVY} !important;
    }}

    /* TEXTO SUPERIOR ESQUERDO (CORRIGINDO BRANCO SOBRE BRANCO) */
    /* Garante que o rótulo do uploader seja legível (Azul Marinho) */
    [data-testid="stSidebar"] label p {{
        color: {HUB_NAVY} !important;
        font-weight: bold !important;
    }}
    
    /* WIDGET DE UPLOAD */
    div[data-testid="stFileUploadDropzone"] {{
        background-color: {WHITE} !important;
        border: 2px dashed {HUB_BLUE} !important;
        border-radius: 10px !important;
    }}
    
    /* Cores de Títulos e Texto Geral */
    h1, h2, h3, h4 {{ color: {HUB_NAVY} !important; }}
    .stMarkdown p {{ color: {HUB_NAVY} !important; }}

    /* CARTÕES DE RELATÓRIO */
    .report-card {{
        background-color: {WHITE} !important;
        border-top: 5px solid {HUB_BLUE} !important;
        padding: 20px !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important;
        color: {HUB_NAVY} !important;
        margin-bottom: 20px !important;
    }}
    
    .data-line {{
        display: flex;
        justify-content: space-between;
        border-bottom: 1px solid #f0f0f0;
        padding: 8px 0;
        font-size: 0.92em;
    }}
    </style>
    """, unsafe_allow_html=True)

# ==========================================================
# 4. MOTOR TÉCNICO (INTEGRIDADE MANTIDA)
# ==========================================================

def butter_bandpass_filter(data, fs=2000):
    nyq = 0.5 * fs
    b, a = butter(4, [6/nyq, 500/nyq], btype='band')
    if len(data) <= (max(len(a), len(b)) * 3): return data
    return filtfilt(b, a, data)

def calculate_rms(data, fs=2000):
    rectified = np.abs(data - np.mean(data))
    window = int(fs * 0.01)
    return np.sqrt(np.convolve(rectified**2, np.ones(window)/window, mode='same'))

def parse_sylk(file):
    try:
        content = file.getvalue().decode("utf-8", errors="ignore")
        data_map = {}
        ch_names = {4: "CH 1", 5: "CH 2"}
        for line in content.splitlines():
            if line.startswith('C;'):
                p = line.split(';')
                try:
                    r, c = int(p[1][1:]), int(p[2][1:])
                    val_raw = p[3]
                    val = float(val_raw[1:].replace('"','')) if val_raw.startswith('K') else float(val_raw.replace('"',''))
                    if r not in data_map: data_map[r] = {}
                    data_map[r][c] = val
                    if r == 4 and c in [4, 5]: ch_names[c] = val_raw[1:].replace('"','')
                except: continue
        df_raw = pd.DataFrame.from_dict(data_map, orient='index').sort_index()
        df = pd.DataFrame({'time': df_raw[1].values, 'CH1': df_raw[4].values, 'CH2': df_raw[5].values}).dropna().iloc[5:]
        return df, [ch_names[4], ch_names[5]]
    except Exception as e:
        st.error(f"Erro: {e}")
        return None, None

# ==========================================================
# 5. UI PRINCIPAL
# ==========================================================

# Cabeçalho e Idioma
h_col1, h_col2 = st.columns([4, 1])
with h_col2:
    sel_lang = st.selectbox("🌐 Language", list(LANGS.keys()))
    tr = LANGS[sel_lang]
with h_col1:
    st.title(tr["title"])

# Upload na Barra Lateral
uploaded_file = st.sidebar.file_uploader(tr["upload_label"], type=["slk", "csv"])

if uploaded_file:
    df_emg, labels = parse_sylk(uploaded_file)
    if df_emg is not None:
        fs = 2000
        onsets_results = {}
        ui_cols = st.columns(2)

        for i, (ch_col, name) in enumerate(zip(['CH1', 'CH2'], labels)):
            with ui_cols[i]:
                st.subheader(name)
                filt = butter_bandpass_filter(df_emg[ch_col].values)
                rms_signal = calculate_rms(filt)
                
                fig = go.Figure(go.Scatter(x=df_emg['time'], y=rms_signal, line=dict(color=HUB_NAVY, width=1.3)))
                fig.update_layout(
                    height=400, margin=dict(l=10, r=10, t=10, b=10),
                    dragmode='select', selectdirection='h',
                    plot_bgcolor=WHITE,
                    paper_bgcolor='rgba(0,0,0,0)',
                    newshape=dict(line=dict(color=HUB_BLUE, width=2), fillcolor=HUB_BLUE, opacity=0.3),
                    xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
                    yaxis=dict(showgrid=True, gridcolor='#f0f0f0')
                )
                
                sel = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key=f"p_{ch_col}_{sel_lang}")

                if sel and "selection" in sel and "box" in sel["selection"] and len(sel["selection"]["box"]) > 0:
                    t1, t2 = sel["selection"]["box"][0]["x"][0], sel["selection"]["box"][0]["x"][1]
                    mask = (df_emg['time'] >= t1) & (df_emg['time'] <= t2)
                    st_time, st_rms = df_emg['time'][mask].values, rms_signal[mask]
                    
                    if len(st_rms) > 10:
                        baseline = rms_signal[:400]
                        thr = np.mean(baseline) + (3 * np.std(baseline))
                        idx = next((j for j in range(len(st_rms)-40) if np.all(st_rms[j:j+40] >= thr)), None)
                        
                        v_max, v_mean, v_area = np.max(st_rms), np.mean(st_rms), simpson(st_rms, dx=1/fs)
                        duration = t2 - t1

                        st.markdown(f"""
                        <div class="report-card">
                            <h4 style="margin:0 0 15px 0; color:{HUB_BLUE}; font-size:1.1em;">{tr['rep']}</h4>
                            <div class="data-line"><span>{tr['interval']}</span><b>{t1:.3f}s - {t2:.3f}s</b></div>
                            <div class="data-line"><span>{tr['duration']}</span><b>{duration:.3f} s</b></div>
                            <div class="data-line"><span>{tr['ons']}</span><b>{st_time[idx] if idx else "N/D"} s</b></div>
                            <div class="data-line"><span>{tr['peak']}</span><b>{v_max:.2f} µV</b></div>
                            <div class="data-line"><span>{tr['mean_rms']}</span><b>{v_mean:.2f} µV</b></div>
                            <div class="data-line"><span>{tr['area']}</span><b>{v_area:.4f} µV.s</b></div>
                            <p style="font-size:0.8em; color:gray; margin-top:15px; border-top:1px dashed #ddd; padding-top:5px;">
                                {tr['thresh']}: {thr:.4f} µV
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        if idx: onsets_results[i] = st_time[idx]
                else:
                    st.info(tr["info"])

        if len(onsets_results) == 2:
            diff = abs(onsets_results[0] - onsets_results[1]) * 1000
            st.success(f"### ⏱️ {tr['sync']}: **{diff:.2f} ms**")
else:
    st.info(tr["wait"])
