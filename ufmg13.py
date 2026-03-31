import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt
from scipy.integrate import simpson

# ==========================================================
# 1. IDENTIDADE HUB ACADEMICA
# ==========================================================
HUB_NAVY = "#001a33"
HUB_BLUE = "#1a73e8"
HUB_BG = "#f0f2f6"
WHITE = "#ffffff"

st.set_page_config(page_title="EMGExpert | Hub Academica", layout="wide")

# ==========================================================
# 2. DICIONÁRIO DE DADOS COMPLETO (RESTAURADO)
# ==========================================================
LANGS = {
    "PORTUGUÊS (BRASILEIRO)": {
        "title": "EMGExpert — Ciência Encontra a Prática",
        "upload": "Carregar arquivo (.slk ou .csv)",
        "info": "🖱️ Selecione uma área no gráfico para análise profunda.",
        "rep": "RELATÓRIO TÉCNICO DETALHADO",
        "interval": "⏱️ Janela de Tempo",
        "duration": "⏳ Duração da Seleção",
        "ons": "🟢 ONSET (Início)",
        "peak": "📈 PICO MÁXIMO (Amplitude)",
        "mean_rms": "🌊 RMS MÉDIO",
        "area": "📊 ÁREA (Integral)",
        "sync": "Diferença de Sincronismo (Delay)",
        "wait": "Aguardando arquivo...",
        "thresh": "Threshold de detecção aplicado"
    }
}

# ==========================================================
# 3. CSS (FOCO EM LEITURA DE DADOS)
# ==========================================================
st.markdown(f"""
    <style>
    .stApp {{ background-color: {HUB_BG} !important; }}
    [data-testid="stSidebar"] {{ background-color: {HUB_NAVY} !important; }}
    
    /* Card de Dados Estilo Tabela */
    .report-card {{
        background-color: {WHITE} !important;
        border-left: 8px solid {HUB_BLUE} !important;
        padding: 20px !important;
        border-radius: 4px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
        color: {HUB_NAVY} !important;
    }}
    .data-line {{
        display: flex;
        justify-content: space-between;
        border-bottom: 1px solid #eee;
        padding: 5px 0;
    }}
    </style>
    """, unsafe_allow_html=True)

# ==========================================================
# 4. FUNÇÕES TÉCNICAS
# ==========================================================

def process_emg(data, fs=2000):
    nyq = 0.5 * fs
    b, a = butter(4, [6/nyq, 500/nyq], btype='band')
    filt = filtfilt(b, a, data)
    rectified = np.abs(filt - np.mean(filt))
    window = int(fs * 0.01)
    return np.sqrt(np.convolve(rectified**2, np.ones(window)/window, mode='same'))

def parse_sylk(file):
    try:
        content = file.getvalue().decode("utf-8", errors="ignore")
        data_map = {}
        names = {4: "CH 1", 5: "CH 2"}
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
        df = pd.DataFrame({'t': df_r[1].values, 'c1': df_r[4].values, 'c2': df_r[5].values}).dropna().iloc[5:]
        return df, [names[4], names[5]]
    except: return None, None

# ==========================================================
# 5. UI E LOGA DE ANÁLISE
# ==========================================================

tr = LANGS["PORTUGUÊS (BRASILEIRO)"]
st.title(tr["title"])

up = st.sidebar.file_uploader(tr["upload"], type=["slk", "csv"])

if up:
    df, labels = parse_sylk(up)
    if df is not None:
        onsets = {}
        ui_cols = st.columns(2)
        
        for i, (col, name) in enumerate(zip(['c1', 'c2'], labels)):
            with ui_cols[i]:
                st.subheader(name)
                rms = process_emg(df[col].values)
                
                fig = go.Figure(go.Scatter(x=df['t'], y=rms, line=dict(color=HUB_NAVY, width=1.5)))
                fig.update_layout(height=380, dragmode='select', selectdirection='h', plot_bgcolor=WHITE, margin=dict(l=0,r=0,t=0,b=0))
                sel = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key=f"p_{col}")

                if sel and "selection" in sel and "box" in sel["selection"] and len(sel["selection"]["box"]) > 0:
                    t1, t2 = sel["selection"]["box"][0]["x"][0], sel["selection"]["box"][0]["x"][1]
                    mask = (df['t'] >= t1) & (df['t'] <= t2)
                    st_t, st_rms = df['t'][mask].values, rms[mask]
                    
                    if len(st_rms) > 10:
                        # Cálculo de Onset robusto
                        baseline = rms[:400]
                        thr = np.mean(baseline) + (3 * np.std(baseline))
                        idx = next((j for j in range(len(st_rms)-40) if np.all(st_rms[j:j+40] >= thr)), None)
                        
                        # NOVOS DADOS ADICIONADOS AQUI:
                        v_max = np.max(st_rms)
                        v_mean = np.mean(st_rms)
                        v_area = simpson(st_rms, dx=1/2000)
                        duration = t2 - t1

                        st.markdown(f"""
                        <div class="report-card">
                            <h4 style="margin:0 0 10px 0;">{tr['rep']}</h4>
                            <div class="data-line"><span>{tr['interval']}</span><b>{t1:.2f}s - {t2:.2f}s</b></div>
                            <div class="data-line"><span>{tr['duration']}</span><b>{duration:.3f} s</b></div>
                            <div class="data-line"><span>{tr['ons']}</span><b>{st_t[idx] if idx else "N/D"} s</b></div>
                            <div class="data-line"><span>{tr['peak']}</span><b>{v_max:.2f} µV</b></div>
                            <div class="data-line"><span>{tr['mean_rms']}</span><b>{v_mean:.2f} µV</b></div>
                            <div class="data-line"><span>{tr['area']}</span><b>{v_area:.4f} µV.s</b></div>
                            <p style="font-size:0.8em; color:gray; margin-top:10px;">{tr['thresh']}: {thr:.4f} µV</p>
                        </div>
                        """, unsafe_allow_html=True)
                        if idx: onsets[i] = st_t[idx]
                else:
                    st.info(tr["info"])

        if len(onsets) == 2:
            diff = abs(onsets[0] - onsets[1]) * 1000
            st.success(f"### ⏱️ {tr['sync']}: **{diff:.2f} ms**")
else:
    st.info(tr["wait"])
