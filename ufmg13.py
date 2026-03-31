import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt
from scipy.integrate import simpson

# ==========================================================
# 1. IDENTIDADE VISUAL HUB ACADEMICA (ATUALIZADA)
# ==========================================================
HUB_NAVY = "#001a33"   # Azul Marinho (Barra lateral e Títulos)
HUB_BLUE = "#1a73e8"   # Azul Botões (Destaque)
HUB_BG = "#f8f9fa"     # Cinza suave para o fundo da página
WHITE = "#ffffff"

st.set_page_config(page_title="EMGExpert | Hub Academica", layout="wide")

# ==========================================================
# 2. DICIONÁRIO DE TRADUÇÕES
# ==========================================================
LANGS = {
    "PORTUGUÊS (BRASILEIRO)": {
        "title": "EMGExpert — Ciência Encontra a Prática",
        "upload": "Escolha o arquivo (.slk ou .csv)",
        "info": "🖱️ Arraste no gráfico para analisar.",
        "rep": "RELATÓRIO TÉCNICO",
        "ons": "ONSET (INÍCIO)",
        "peak": "PICO MÁXIMO",
        "area": "ÁREA (INTEGRAL)",
        "sync": "Diferença de Sincronismo (Delay)",
        "wait": "Aguardando upload do arquivo...",
        "thresh": "Threshold de detecção"
    },
    "ENGLISH": { "title": "EMGExpert — Science Meets Practice", "upload": "Choose file", "info": "🖱️ Drag to analyze.", "rep": "TECHNICAL REPORT", "ons": "ONSET", "peak": "PEAK", "area": "AREA", "sync": "Delay", "wait": "Waiting...", "thresh": "Threshold" },
    "ESPAÑOL": { "title": "EMGExpert — Ciencia y Práctica", "upload": "Cargar archivo", "info": "🖱️ Arrastre para analizar.", "rep": "INFORME TÉCNICO", "ons": "ONSET", "peak": "PICO", "area": "ÁREA", "sync": "Delay", "wait": "Esperando...", "thresh": "Umbral" },
    "CHINESE (SIMPLIFIED)": { "title": "EMGExpert — 科学与实践", "upload": "上传文件", "info": "🖱️ 拖动进行分析", "rep": "技术报告", "ons": "起始点", "peak": "最大峰值", "area": "面积", "sync": "延迟", "wait": "等待中...", "thresh": "阈值" }
}

# ==========================================================
# 3. CSS PARA CORRIGIR O BOTÃO E O FUNDO (PALETA DO SITE)
# ==========================================================
st.markdown(f"""
    <style>
    /* Fundo da Página Central */
    .stApp {{ background-color: {HUB_BG}; }}
    
    /* Barra Lateral */
    [data-testid="stSidebar"] {{ background-color: {HUB_NAVY}; }}
    [data-testid="stSidebar"] * {{ color: {WHITE} !important; }}
    
    /* BOTÃO DE UPLOAD - CORREÇÃO DE VISIBILIDADE */
    section[data-testid="stFileUploadDropzone"] {{
        background-color: {WHITE};
        border: 2px dashed {HUB_BLUE};
        border-radius: 10px;
    }}
    label[data-testid="stWidgetLabel"] {{ color: {HUB_NAVY} !important; font-weight: bold; }}
    
    /* Estilização dos Cartões de Dados */
    .report-card {{
        background-color: {WHITE};
        border-top: 5px solid {HUB_BLUE};
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        color: {HUB_NAVY};
        margin-top: 15px;
    }}

    /* Títulos Principais */
    h1, h2, h3, h4 {{ 
        color: {HUB_NAVY} !important; 
        font-family: 'Inter', sans-serif; 
        font-weight: 700 !important; 
    }}
    
    /* Ajuste no seletor de linguagem */
    .stSelectbox div[data-baseweb="select"] {{
        background-color: {WHITE};
        border-radius: 5px;
    }}
    </style>
    """, unsafe_allow_html=True)

# ==========================================================
# 4. FUNÇÕES TÉCNICAS
# ==========================================================

def butter_filter(data, fs=2000):
    nyq = 0.5 * fs
    b, a = butter(4, [6/nyq, 500/nyq], btype='band')
    y = filtfilt(b, a, data)
    return np.abs(y - np.mean(y))

def calculate_rms(rect, fs=2000):
    win = int(fs * 0.01)
    return np.sqrt(np.convolve(rect**2, np.ones(win)/win, mode='same'))

def parse_sylk(file):
    try:
        content = file.getvalue().decode("utf-8", errors="ignore")
        data_map = {{}}
        names = {{4: "CH 1", 5: "CH 2"}}
        for line in content.splitlines():
            if line.startswith('C;'):
                p = line.split(';')
                try:
                    r, c = int(p[1][1:]), int(p[2][1:])
                    val = float(p[3][1:].replace('"','')) if p[3].startswith('K') else float(p[3].replace('"',''))
                    if r not in data_map: data_map[r] = {{}}
                    data_map[r][c] = val
                    if r == 4 and c in [4, 5]: names[c] = p[3][1:].replace('"','')
                except: continue
        df_r = pd.DataFrame.from_dict(data_map, orient='index').sort_index()
        df = pd.DataFrame({{'time': df_r[1].values, 'CH1': df_r[4].values, 'CH2': df_r[5].values}}).dropna().iloc[5:]
        return df, [names[4], names[5]]
    except: return None, None

# ==========================================================
# 5. INTERFACE
# ==========================================================

col_t, col_l = st.columns([4, 1])
with col_l:
    sel_lang = st.selectbox("🌐 Language", list(LANGS.keys()))
    tr = LANGS[sel_lang]
with col_t:
    st.title(tr["title"])

up_file = st.sidebar.file_uploader(tr["upload"], type=["slk", "csv"])

if up_file:
    df_e, labels = parse_sylk(up_file)
    if df_e is not None:
        onsets = {{}}
        ui_c = st.columns(2)
        for i, (ch, name) in enumerate(zip(['CH1', 'CH2'], labels)):
            with ui_c[i]:
                st.subheader(name)
                rms = calculate_rms(butter_filter(df_e[ch].values))
                
                fig = go.Figure(go.Scatter(x=df_e['time'], y=rms, line=dict(color=HUB_NAVY, width=1.5)))
                fig.update_layout(
                    height=400, dragmode='select', selectdirection='h',
                    plot_bgcolor=WHITE, paper_bgcolor='rgba(0,0,0,0)',
                    newshape=dict(line=dict(color=HUB_BLUE, width=2), fillcolor=HUB_BLUE, opacity=0.25),
                    margin=dict(l=10, r=10, t=10, b=10)
                )
                
                s = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key=f"p_{ch}_{sel_lang}")

                if s and "selection" in s and "box" in s["selection"] and len(s["selection"]["box"]) > 0:
                    t1, t2 = s["selection"]["box"][0]["x"][0], s["selection"]["box"][0]["x"][1]
                    mask = (df_e['time'] >= t1) & (df_e['time'] <= t2)
                    s_t, s_r = df_e['time'][mask].values, rms[mask]
                    
                    if len(s_r) > 10:
                        v_max, area = np.max(s_r), simpson(s_r, dx=1/2000)
                        baseline = rms[:400]
                        thr = np.mean(baseline) + (3 * np.std(baseline))
                        
                        # Buscando Onset no segmento
                        idx = None
                        for j in range(len(s_r)-40):
                            if np.all(s_r[j:j+40] >= thr):
                                idx = j; break

                        st.markdown(f"""
                        <div class="report-card">
                            <b style="color:{HUB_BLUE}; font-size:1.1em;">{tr['rep']}</b><br><br>
                            • <b>{tr['ons']}:</b> {s_t[idx] if idx else "N/D"} s<br>
                            • <b>{tr['peak']}:</b> {v_max:.2f} µV<br>
                            • <b>{tr['area']}:</b> {area:.4f} µV.s<br>
                            <hr style="border:0.5px solid #eee">
                            <small>{tr['thresh']}: {thr:.4f} µV</small>
                        </div>
                        """, unsafe_allow_html=True)
                        if idx: onsets[i] = s_t[idx]
                else: st.info(tr["info"])

        if len(onsets) == 2:
            st.success(f"### ⏱️ {tr['sync']}: {abs(onsets[0]-onsets[1])*1000:.2f} ms")
else:
    st.info(tr["wait"])
