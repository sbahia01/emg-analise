import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt
from scipy.integrate import simpson
import io

# ==========================================================
# 1. IDENTIDADE VISUAL HUB ACADEMICA
# ==========================================================
HUB_NAVY = "#001a33"   
HUB_BLUE = "#1a73e8"   
HUB_BG = "#eef4ff"     
WHITE = "#ffffff"

st.set_page_config(page_title="EMGExpert | Hub Academica", layout="wide")

# ==========================================================
# 2. DICIONÁRIO DE TRADUÇÕES COMPLETO
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
        "thresh": "Threshold de detecção aplicado",
        "dl_btn": "📥 Baixar Relatório Completo"
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
        "thresh": "Threshold",
        "dl_btn": "📥 Download Full Report"
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
        "thresh": "Umbral",
        "dl_btn": "📥 Descargar Informe"
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
        "thresh": "阈值",
        "dl_btn": "📥 下载报告"
    }
}

# ==========================================================
# 3. CSS CUSTOMIZADO (BOTÕES ESTILO EXPLORE TOOLS)
# ==========================================================
st.markdown(f"""
    <style>
    .stApp {{ background-color: {HUB_BG} !important; }}
    [data-testid="stSidebar"] {{ background-color: {HUB_NAVY} !important; }}

    /* TEXTOS BARRA LATERAL */
    [data-testid="stSidebar"] label p, [data-testid="stSidebar"] p, .stFileUploaderFileName {{
        color: {WHITE} !important;
    }}
    
    /* SELETOR DE IDIOMAS E BOTÃO DE DOWNLOAD (ESTILO EXPLORE TOOLS) */
    div[data-testid="stSelectbox"] div[data-baseweb="select"], 
    .stDownloadButton button {{
        background: linear-gradient(90deg, #1a73e8 0%, #00d4ff 100%) !important;
        border-radius: 50px !important;
        border: none !important;
        color: {WHITE} !important;
        padding: 10px 25px !important;
        box-shadow: 0 4px 15px rgba(26, 115, 232, 0.4) !important;
        transition: 0.3s !important;
        font-weight: bold !important;
        width: auto !important;
    }}
    .stDownloadButton button:hover {{
        transform: scale(1.05) !important;
        box-shadow: 0 6px 20px rgba(26, 115, 232, 0.6) !important;
    }}

    /* TÍTULOS E CARTÕES */
    h1, h2, h3, h4 {{ color: {HUB_NAVY} !important; font-weight: 800 !important; }}
    .report-card {{
        background-color: {WHITE} !important;
        border-top: 6px solid {HUB_BLUE} !important;
        padding: 25px !important;
        border-radius: 12px !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08) !important;
        color: {HUB_NAVY} !important;
        margin-bottom: 25px !important;
    }}
    .data-line {{
        display: flex;
        justify-content: space-between;
        border-bottom: 1px solid #f2f2f2;
        padding: 10px 0;
    }}
    </style>
    """, unsafe_allow_html=True)

# ==========================================================
# 4. MOTOR TÉCNICO
# ==========================================================

def butter_bandpass_filter(data, fs=2000):
    nyq = 0.5 * fs
    b, a = butter(4, [6/nyq, 500/nyq], btype='band')
    return filtfilt(b, a, data) if len(data) > 12 else data

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
                r, c = int(p[1][1:]), int(p[2][1:])
                val_raw = p[3]
                val = float(val_raw[1:].replace('"','')) if val_raw.startswith('K') else float(val_raw.replace('"',''))
                if r not in data_map: data_map[r] = {}
                data_map[r][c] = val
                # CAPTURA O NOME REAL DOS CANAIS
                if r == 4 and c in [4, 5]: ch_names[c] = val_raw[1:].replace('"','')
        df_raw = pd.DataFrame.from_dict(data_map, orient='index').sort_index()
        df = pd.DataFrame({'time': df_raw[1].values, 'CH1': df_raw[4].values, 'CH2': df_raw[5].values}).dropna().iloc[5:]
        return df, [ch_names[4], ch_names[5]]
    except: return None, None

# ==========================================================
# 5. UI PRINCIPAL
# ==========================================================

h_col1, h_col2 = st.columns([3, 1])
with h_col2:
    sel_lang = st.selectbox("Lang", list(LANGS.keys()), label_visibility="collapsed")
    tr = LANGS[sel_lang]
with h_col1:
    st.title(tr["title"])

uploaded_file = st.sidebar.file_uploader(tr["upload_label"], type=["slk", "csv"])

if uploaded_file:
    df_emg, labels = parse_sylk(uploaded_file)
    if df_emg is not None:
        onsets_results = {}
        full_stats = []
        ui_cols = st.columns(2)

        for i, (ch_col, name) in enumerate(zip(['CH1', 'CH2'], labels)):
            with ui_cols[i]:
                st.subheader(f"📊 {name}") # EXIBE NOME REAL
                filt = butter_bandpass_filter(df_emg[ch_col].values)
                rms_signal = calculate_rms(filt)
                
                fig = go.Figure(go.Scatter(x=df_emg['time'], y=rms_signal, line=dict(color=WHITE, width=1.3)))
                fig.update_layout(
                    height=400, margin=dict(l=10, r=10, t=10, b=10),
                    dragmode='select', selectdirection='h',
                    plot_bgcolor=HUB_NAVY, paper_bgcolor=HUB_NAVY,
                    newshape=dict(line=dict(color=HUB_BLUE, width=2), fillcolor=HUB_BLUE, opacity=0.3),
                    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color=WHITE)),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color=WHITE))
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
                        
                        stats = {
                            "name": name, "interval": f"{t1:.3f}s - {t2:.3f}s",
                            "dur": t2-t1, "ons": st_time[idx] if idx else None,
                            "peak": np.max(st_rms), "mean": np.mean(st_rms),
                            "area": simpson(st_rms, dx=1/2000)
                        }
                        full_stats.append(stats)
                        if stats["ons"]: onsets_results[i] = stats["ons"]

                        st.markdown(f"""
                        <div class="report-card">
                            <h4 style="margin:0 0 15px 0; color:{HUB_BLUE};">{tr['rep']}</h4>
                            <div class="data-line"><span>{tr['interval']}</span><b>{stats['interval']}</b></div>
                            <div class="data-line"><span>{tr['duration']}</span><b>{stats['dur']:.3f} s</b></div>
                            <div class="data-line"><span>{tr['ons']}</span><b>{stats['ons'] if stats['ons'] else "N/D"} s</b></div>
                            <div class="data-line"><span>{tr['peak']}</span><b>{stats['peak']:.2f} µV</b></div>
                            <div class="data-line"><span>{tr['mean_rms']}</span><b>{stats['mean']:.2f} µV</b></div>
                            <div class="data-line"><span>{tr['area']}</span><b>{stats['area']:.4f} µV.s</b></div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info(tr["info"])

        # COMPARATIVO FINAL E BOTÃO DE DOWNLOAD
        if len(full_stats) > 0:
            st.divider()
            col_res1, col_res2 = st.columns([2, 1])
            
            with col_res1:
                if len(onsets_results) == 2:
                    diff = abs(onsets_results[0] - onsets_results[1]) * 1000
                    st.success(f"### ⏱️ {tr['sync']}: **{diff:.2f} ms**")
            
            with col_res2:
                # GERA CSV PARA DOWNLOAD
                report_df = pd.DataFrame(full_stats)
                csv = report_df.to_csv(index=False).encode('utf-8')
                st.download_button(tr["dl_btn"], data=csv, file_name="EMG_Report.csv", mime="text/csv")
else:
    st.info(tr["wait"])
