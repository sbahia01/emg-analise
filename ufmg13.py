import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt
from scipy.integrate import simpson

# ==========================================================
# 1. CONFIGURAÇÃO E CORES (HUB ACADEMICA)
# ==========================================================
HUB_NAVY = "#001a33"
HUB_BLUE = "#1a73e8"
HUB_BG = "#f0f2f6"
WHITE = "#ffffff"

st.set_page_config(page_title="EMGExpert | Hub Academica", layout="wide")

# ==========================================================
# 2. DICIONÁRIO EXPANDIDO (VOLTANDO COM TODAS AS LINHAS)
# ==========================================================
LANGS = {
    "PORTUGUÊS (BRASILEIRO)": {
        "title": "EMGExpert — Ciência Encontra a Prática",
        "upload": "Selecione o arquivo (.slk ou .csv)",
        "info": "🖱️ Arraste no gráfico para analisar a contração.",
        "rep": "RELATÓRIO TÉCNICO",
        "interval": "⏱️ Intervalo Selecionado",
        "ons": "🟢 ONSET (INÍCIO)",
        "peak": "📈 PICO MÁXIMO",
        "area": "📊 ÁREA (INTEGRAL)",
        "sync": "Diferença de Sincronismo (Delay)",
        "wait": "Aguardando upload do arquivo...",
        "thresh": "Threshold de detecção"
    },
    "ENGLISH": {
        "title": "EMGExpert — Science Meets Practice",
        "upload": "Choose file",
        "info": "🖱️ Drag to analyze.",
        "rep": "TECHNICAL REPORT",
        "interval": "⏱️ Selected Interval",
        "ons": "🟢 ONSET",
        "peak": "📈 PEAK AMPLITUDE",
        "area": "📊 AREA (INTEGRAL)",
        "sync": "Sync Delay",
        "wait": "Waiting...",
        "thresh": "Threshold"
    }
}

# ==========================================================
# 3. CSS (VISIBILIDADE E ESTILO)
# ==========================================================
st.markdown(f"""
    <style>
    .stApp {{ background-color: {HUB_BG} !important; }}
    [data-testid="stSidebar"] {{ background-color: {HUB_NAVY} !important; }}
    [data-testid="stSidebar"] * {{ color: {WHITE} !important; }}
    
    /* Botão de Upload Corrigido */
    section[data-testid="stFileUploadDropzone"] {{
        background-color: {WHITE} !important;
        border: 2px dashed {HUB_BLUE} !important;
        border-radius: 10px !important;
    }}
    
    .report-card {{
        background-color: {WHITE} !important;
        border-top: 5px solid {HUB_BLUE} !important;
        padding: 20px !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05) !important;
        color: {HUB_NAVY} !important;
        line-height: 1.6 !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# ==========================================================
# 4. PROCESSAMENTO
# ==========================================================

def butter_filter(data, fs=2000):
    nyq = 0.5 * fs
    b, a = butter(4, [6/nyq, 500/nyq], btype='band')
    return filtfilt(b, a, data)

def calculate_rms(data, fs=2000):
    rectified = np.abs(data - np.mean(data))
    window = int(fs * 0.01)
    return np.sqrt(np.convolve(rectified**2, np.ones(window)/window, mode='same'))

def parse_sylk(file):
    try:
        content = file.getvalue().decode("utf-8", errors="ignore")
        data_map = {}
        names = {4: "Canal 1", 5: "Canal 2"}
        for line in content.splitlines():
            if line.startswith('C;'):
                p = line.split(';')
                try:
                    r, c = int(p[1][1:]), int(p[2][1:])
                    v = p[3]
                    val = float(v[1:].replace('"','')) if v.startswith('K') else float(v.replace('"',''))
                    if r not in data_map: data_map[r] = {}
                    data_map[r][c] = val
                    if r == 4 and c in [4, 5]: names[c] = v[1:].replace('"','')
                except: continue
        df_r = pd.DataFrame.from_dict(data_map, orient='index').sort_index()
        df = pd.DataFrame({'time': df_r[1].values, 'CH1': df_r[4].values, 'CH2': df_r[5].values}).dropna().iloc[5:]
        return df, [names[4], names[5]]
    except: return None, None

# ==========================================================
# 5. INTERFACE
# ==========================================================

header_col1, header_col2 = st.columns([4, 1])
with header_col2:
    sel_lang = st.selectbox("🌐 Language", list(LANGS.keys()))
    tr = LANGS[sel_lang]
with header_col1:
    st.title(tr["title"])

uploaded_file = st.sidebar.file_uploader(tr["upload"], type=["slk", "csv"])

if uploaded_file:
    df, labels = parse_sylk(uploaded_file)
    if df is not None:
        onsets = {}
        cols = st.columns(2)
        for i, (ch, name) in enumerate(zip(['CH1', 'CH2'], labels)):
            with cols[i]:
                st.subheader(name)
                rms = calculate_rms(butter_filter(df[ch].values))
                
                fig = go.Figure(go.Scatter(x=df['time'], y=rms, line=dict(color=HUB_NAVY, width=1.2)))
                fig.update_layout(height=400, dragmode='select', selectdirection='h', plot_bgcolor=WHITE, margin=dict(l=0,r=0,t=0,b=0))
                
                sel = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key=f"p_{ch}_{sel_lang}")

                if sel and "selection" in sel and "box" in sel["selection"] and len(sel["selection"]["box"]) > 0:
                    t1, t2 = sel["selection"]["box"][0]["x"][0], sel["selection"]["box"][0]["x"][1]
                    mask = (df['time'] >= t1) & (df['time'] <= t2)
                    s_t, s_r = df['time'][mask].values, rms[mask]
                    
                    if len(s_r) > 10:
                        baseline = rms[:400]
                        thr = np.mean(baseline) + (3 * np.std(baseline))
                        idx = next((j for j in range(len(s_r)-40) if np.all(s_r[j:j+40] >= thr)), None)
                        
                        v_max, area = np.max(s_r), simpson(s_r, dx=1/2000)
                        
                        # RELATÓRIO COMPLETO (VOLTANDO COM TODAS AS LINHAS)
                        st.markdown(f"""
                        <div class="report-card">
                            <h4 style="margin:0 0 15px 0; color:{HUB_BLUE}; border-bottom:1px solid #eee;">{tr['rep']}</h4>
                            • <b>{tr['interval']}:</b> {t1:.3f}s — {t2:.3f}s<br>
                            • <b>{tr['ons']}:</b> {s_t[idx] if idx else "N/D"} s<br>
                            • <b>{tr['peak']}:</b> {v_max:.2f} µV<br>
                            • <b>{tr['area']}:</b> {area:.4f} µV.s<br>
                            <div style="margin-top:10px; padding-top:10px; border-top:1px dashed #ddd; font-size:0.85em; color:#666;">
                                {tr['thresh']}: {thr:.4f} µV
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        if idx: onsets[i] = s_t[idx]
                else:
                    st.info(tr["info"])

        if len(onsets) == 2:
            st.success(f"### ⏱️ {tr['sync']}: **{abs(onsets[0]-onsets[1])*1000:.2f} ms**")
else:
    st.info(tr["wait"])
