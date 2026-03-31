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
HUB_BG = "#f0f2f6"     # Cinza de fundo (Contraste)
WHITE = "#ffffff"

st.set_page_config(page_title="EMGExpert | Hub Academica", layout="wide")

# ==========================================================
# 2. DICIONÁRIO DE TRADUÇÕES COMPLETO (RESTAURADO + AMPLIADO)
# ==========================================================
LANGS = {
    "PORTUGUÊS (BRASILEIRO)": {
        "title": "EMGExpert — Ciência Encontra a Prática",
        "upload": "Carregar arquivo (.slk ou .csv)",
        "info": "🖱️ Arraste no gráfico para analisar a contração.",
        "rep": "RELATÓRIO TÉCNICO",
        "interval": "⏱️ JANELA DE TEMPO",
        "duration": "⏳ DURAÇÃO DA SELEÇÃO",
        "ons": "🟢 ONSET (INÍCIO)",
        "peak": "📈 PICO MÁXIMO",
        "mean_rms": "🌊 RMS MÉDIO",
        "area": "📊 ÁREA (INTEGRAL)",
        "sync": "Diferença de Sincronismo (Delay)",
        "wait": "Aguardando upload do arquivo...",
        "thresh": "Threshold de detecção"
    },
    "ENGLISH": {
        "title": "EMGExpert — Science Meets Practice",
        "upload": "Upload file (.slk or .csv)",
        "info": "🖱️ Drag on the chart to analyze contraction.",
        "rep": "TECHNICAL REPORT",
        "interval": "⏱️ TIME WINDOW",
        "duration": "⏳ SELECTION DURATION",
        "ons": "🟢 ONSET",
        "peak": "📈 PEAK AMPLITUDE",
        "mean_rms": "🌊 MEAN RMS",
        "area": "📊 AREA (INTEGRAL)",
        "sync": "Synchronization Delay",
        "wait": "Waiting for file upload...",
        "thresh": "Detection Threshold"
    },
    "ESPAÑOL": {
        "title": "EMGExpert — La Ciencia se Encuentra con la Práctica",
        "upload": "Cargar archivo (.slk o .csv)",
        "info": "🖱️ Arrastre en el gráfico para analizar.",
        "rep": "INFORME TÉCNICO",
        "interval": "⏱️ VENTANA DE TIEMPO",
        "duration": "⏳ DURACIÓN",
        "ons": "🟢 COMIENZO (ONSET)",
        "peak": "📈 PICO MÁXIMO",
        "mean_rms": "🌊 RMS MEDIO",
        "area": "📊 ÁREA (INTEGRAL)",
        "sync": "Diferencia de Sincronismo",
        "wait": "Esperando carga de archivo...",
        "thresh": "Umbral de detección"
    },
    "CHINESE (SIMPLIFIED)": {
        "title": "EMGExpert — 科学与实践的结合",
        "upload": "上传文件 (.slk 或 .csv)",
        "info": "🖱️ 在图表上拖动以进行分析。",
        "rep": "技术报告",
        "interval": "⏱️ 时间窗口",
        "duration": "⏳ 持续时间",
        "ons": "🟢 起始点 (Onset)",
        "peak": "📈 最大峰值",
        "mean_rms": "🌊 平均均方根 (RMS)",
        "area": "📊 面积 (积分)",
        "sync": "同步延迟",
        "wait": "等待文件上传...",
        "thresh": "检测阈值"
    }
}

# ==========================================================
# 3. ESTILIZAÇÃO CSS (INTEGRIDADE VISUAL)
# ==========================================================
st.markdown(f"""
    <style>
    .stApp {{ background-color: {HUB_BG} !important; }}
    
    [data-testid="stSidebar"] {{
        background-color: {HUB_NAVY} !important;
        color: {WHITE} !important;
    }}
    [data-testid="stSidebar"] * {{ color: {WHITE} !important; }}
    
    section[data-testid="stFileUploadDropzone"] {{
        background-color: {WHITE} !important;
        border: 2px dashed {HUB_BLUE} !important;
        border-radius: 10px !important;
    }}

    h1, h2, h3 {{ color: {HUB_NAVY} !important; font-weight: 800 !important; }}

    .report-card {{
        background-color: {WHITE} !important;
        border-top: 5px solid {HUB_BLUE} !important;
        padding: 20px !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1) !important;
        color: {HUB_NAVY} !important;
        margin-bottom: 20px !important;
    }}
    
    .data-line {{
        display: flex;
        justify-content: space-between;
        border-bottom: 1px solid #eee;
        padding: 8px 0;
        font-size: 0.95em;
    }}
    </style>
    """, unsafe_allow_html=True)

# ==========================================================
# 4. MOTOR TÉCNICO (ROBUSTEZ TOTAL)
# ==========================================================

def butter_bandpass_filter(data, fs=2000):
    nyq = 0.5 * fs
    b, a = butter(4, [6/nyq, 500/nyq], btype='band')
    if len(data) <= 12: return data
    return filtfilt(b, a, data)

def calculate_rms(data, fs=2000):
    rectified = np.abs(data - np.mean(data))
    window = int(fs * 0.01) # 10ms
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
        df = pd.DataFrame({
            'time': df_raw[1].values,
            'CH1': df_raw[4].values,
            'CH2': df_raw[5].values
        }).dropna().iloc[5:]
        return df, [ch_names[4], ch_names[5]]
    except Exception as e:
        st.error(f"Erro: {e}")
        return None, None

# ==========================================================
# 5. INTERFACE DO USUÁRIO
# ==========================================================

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
                filt = butter_bandpass_filter(df_emg[ch_col].values)
                rms_signal = calculate_rms(filt)
                
                fig = go.Figure(go.Scatter(x=df_emg['time'], y=rms_signal, line=dict(color=HUB_NAVY, width=1.3)))
                fig.update_layout(
                    height=400, margin=dict(l=10, r=10, t=10, b=10),
                    dragmode='select', selectdirection='h',
                    plot_bgcolor=WHITE,
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
                        # Cálculos Técnicos
                        baseline = rms_signal[:400]
                        thr = np.mean(baseline) + (3 * np.std(baseline))
                        idx = next((j for j in range(len(st_rms)-40) if np.all(st_rms[j:j+40] >= thr)), None)
                        
                        v_max = np.max(st_rms)
                        v_mean = np.mean(st_rms)
                        v_area = simpson(st_rms, dx=1/fs)
                        duration = t2 - t1

                        # Relatório Integral sem cortes
                        st.markdown(f"""
                        <div class="report-card">
                            <h4 style="margin:0 0 15px 0; color:{HUB_BLUE};">{tr['rep']}</h4>
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
