import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt
from scipy.integrate import simpson
import io
import time

# ==============================================================================
# 1. CONFIGURAÇÕES DE PÁGINA E IDENTIDADE VISUAL (HUB ACADEMICA)
# ==============================================================================
# Definição de paleta de cores de alto contraste para ambiente clínico/acadêmico
HUB_NAVY = "#001a33"   # Azul Marinho (Sidebar e Background de Gráficos)
HUB_BLUE = "#1a73e8"   # Azul Royal (Botões e Destaques)
HUB_LIGHT_BLUE = "#00d4ff" # Ciano para Gradientes
HUB_BG = "#eef4ff"     # Background Principal
WHITE = "#ffffff"
GRAY_TEXT = "#666666"

st.set_page_config(
    page_title="EMGExpert Professional | Hub Academica",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. SISTEMA MULTILINGUE (DICIONÁRIO COMPLETO)
# ==============================================================================
LANGS = {
    "PORTUGUÊS (BRASILEIRO)": {
        "title": "EMGExpert — Ciência Encontra a Prática",
        "header_desc": "Análise Eletromiográfica de Alta Precisão para Pesquisa e Clínica",
        "upload_label": "📂 SELECIONE O ARQUIVO DE DADOS (.SLK OU .CSV)",
        "info": "🖱️ Instrução: Arraste o mouse sobre o gráfico para definir a janela de análise.",
        "rep": "RELATÓRIO TÉCNICO DE CONTRAÇÃO",
        "interval": "⏱️ JANELA SELECIONADA",
        "duration": "⏳ DURAÇÃO DA SELEÇÃO",
        "ons": "🟢 ONSET (INÍCIO DA ATIVAÇÃO)",
        "peak": "📈 PICO MÁXIMO DE AMPLITUDE",
        "mean_rms": "🌊 RMS MÉDIO (VOLTAGEM)",
        "area": "📊 ÁREA SOB A CURVA (INTEGRAL)",
        "sync": "DIFERENÇA DE SINCRONISMO (DELAY ENTRE CANAIS)",
        "wait": "Aguardando carregamento de dados na barra lateral...",
        "thresh": "Threshold (Limiar) de detecção calculado",
        "dl_btn": "📥 EXPORTAR RELATÓRIO COMPLETO (.CSV)",
        "sidebar_title": "CONFIGURAÇÕES",
        "footer_msg": "Desenvolvido para Hub Academica © 2026"
    },
    "ENGLISH": {
        "title": "EMGExpert — Science Meets Practice",
        "header_desc": "High-Precision Electromyographic Analysis for Research & Clinic",
        "upload_label": "📂 SELECT DATA FILE (.SLK OR .CSV)",
        "info": "🖱️ Instruction: Drag over the chart to define the analysis window.",
        "rep": "TECHNICAL CONTRACTION REPORT",
        "interval": "⏱️ SELECTED WINDOW",
        "duration": "⏳ SELECTION DURATION",
        "ons": "🟢 ONSET (ACTIVATION START)",
        "peak": "📈 PEAK AMPLITUDE",
        "mean_rms": "🌊 MEAN RMS (VOLTAGE)",
        "area": "📊 AREA UNDER CURVE (INTEGRAL)",
        "sync": "SYNCHRONIZATION DIFFERENCE (CHANNEL DELAY)",
        "wait": "Waiting for data upload in the sidebar...",
        "thresh": "Calculated detection threshold",
        "dl_btn": "📥 EXPORT FULL REPORT (.CSV)",
        "sidebar_title": "SETTINGS",
        "footer_msg": "Developed for Hub Academica © 2026"
    },
    "ESPAÑOL": {
        "title": "EMGExpert — Ciencia y Práctica",
        "header_desc": "Análisis Electromiográfico de Alta Precisión",
        "upload_label": "📂 SELECCIONE EL ARCHIVO (.SLK O .CSV)",
        "info": "🖱️ Arrastre sobre el gráfico para analizar.",
        "rep": "INFORME TÉCNICO",
        "ons": "🟢 COMIENZO (ONSET)",
        "peak": "📈 PICO MÁXIMO",
        "mean_rms": "🌊 RMS MEDIO",
        "area": "📊 ÁREA (INTEGRAL)",
        "sync": "DIFERENCIA DE SINCRONISMO",
        "wait": "Esperando carga de datos...",
        "thresh": "Umbral de detección",
        "dl_btn": "📥 DESCARGAR INFORME"
    },
    "CHINESE (SIMPLIFIED)": {
        "title": "EMGExpert — 科学与实践",
        "header_desc": "高精度肌电图分析系统",
        "upload_label": "📂 选择文件 (.SLK 或 .CSV)",
        "info": "🖱️ 在图表上拖动以选择分析区域",
        "rep": "技术分析报告",
        "ons": "起始点 (ONSET)",
        "peak": "波峰峰值",
        "mean_rms": "均方根 (RMS)",
        "area": "积分面积",
        "sync": "同步延迟",
        "wait": "等待文件上传...",
        "thresh": "计算阈值",
        "dl_btn": "📥 导出报告"
    }
}

# ==============================================================================
# 3. INJEÇÃO DE CSS (CUSTOMIZAÇÃO AGRESSIVA DE INTERFACE)
# ==============================================================================
st.markdown(f"""
    <style>
    /* 3.1 Estilização do Fundo e Sidebar */
    .stApp {{ background-color: {HUB_BG} !important; }}
    [data-testid="stSidebar"] {{ background-color: {HUB_NAVY} !important; min-width: 350px !important; }}
    
    /* 3.2 Estilização Unificada de Botões (O "Browse Files" incluso aqui) */
    /* Este seletor atinge o botão 'Browse Files', o Selectbox e o Download Button */
    div[data-testid="stSelectbox"] div[data-baseweb="select"], 
    div.stDownloadButton > button,
    section[data-testid="stFileUploader"] button {{
        background: linear-gradient(135deg, {HUB_BLUE} 0%, {HUB_LIGHT_BLUE} 100%) !important;
        border-radius: 50px !important;
        border: none !important;
        color: {WHITE} !important;
        padding: 12px 30px !important;
        font-weight: 800 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        box-shadow: 0 4px 15px rgba(26, 115, 232, 0.4) !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
    }}

    /* 3.3 Efeitos de Interação */
    section[data-testid="stFileUploader"] button:hover, 
    div.stDownloadButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(26, 115, 232, 0.6) !important;
    }}

    /* 3.4 Textos da Sidebar */
    [data-testid="stSidebar"] label p, 
    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] .stFileUploaderFileName {{
        color: {WHITE} !important;
        font-family: 'Inter', sans-serif !important;
    }}

    /* 3.5 Títulos e Tipografia */
    h1, h2, h3, h4 {{ font-family: 'Inter', sans-serif !important; color: {HUB_NAVY} !important; }}
    
    /* 3.6 Cartões de Relatório (Report Cards) */
    .report-card {{
        background-color: {WHITE} !important;
        border-left: 8px solid {HUB_BLUE} !important;
        padding: 25px !important;
        border-radius: 15px !important;
        box-shadow: 0 12px 40px rgba(0,0,0,0.06) !important;
        margin-bottom: 25px !important;
        transition: 0.3s;
    }}
    .report-card:hover {{ transform: scale(1.01); }}
    
    .data-line {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #f0f4f8;
        padding: 12px 0;
    }}
    .label-text {{ font-weight: 600; color: #444; }}
    .value-text {{ font-weight: 800; color: {HUB_BLUE}; font-size: 1.1em; }}
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 4. FUNÇÕES DO MOTOR TÉCNICO (DSP - PROCESSAMENTO DE SINAL)
# ==============================================================================

def butter_bandpass_filter(data, fs=2000, lowcut=6.0, highcut=500.0, order=4):
    """Aplica Filtro Butterworth de 4ª ordem para remover ruídos e artefatos."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    # Proteção para arrays muito curtos
    if len(data) <= (max(len(a), len(b)) * 3):
        return data
    return filtfilt(b, a, data)

def calculate_rms(data, fs=2000, window_ms=10):
    """Calcula o Root Mean Square (RMS) com janela móvel de 10ms."""
    rectified = np.abs(data - np.mean(data))
    window_size = int(fs * (window_ms / 1000))
    if window_size < 1: window_size = 1
    return np.sqrt(np.convolve(rectified**2, np.ones(window_size)/window_size, mode='same'))

def parse_sylk(file):
    """Parser avançado para arquivos SYLK (.slk) do Miotec."""
    try:
        content = file.getvalue().decode("utf-8", errors="ignore")
        data_map = {}
        ch_names = {4: "Canal 1", 5: "Canal 2"} # Default
        
        for line in content.splitlines():
            if line.startswith('C;'):
                parts = line.split(';')
                try:
                    r = int(parts[1][1:]) # Row
                    c = int(parts[2][1:]) # Column
                    val_raw = parts[3]
                    
                    # Extração de valores (Numéricos 'K' ou Strings)
                    if val_raw.startswith('K'):
                        val = float(val_raw[1:].replace('"', ''))
                    else:
                        val = val_raw.replace('"', '')
                        try: val = float(val)
                        except: pass
                    
                    if r not in data_map: data_map[r] = {}
                    data_map[r][c] = val
                    
                    # Identificação de Nomes Reais (Linha 4, Colunas 4 e 5)
                    if r == 4 and c in [4, 5]:
                        ch_names[c] = str(val)
                except:
                    continue

        df_raw = pd.DataFrame.from_dict(data_map, orient='index').sort_index()
        # Coluna 1 = Tempo, Coluna 4 = Canal A, Coluna 5 = Canal B
        df = pd.DataFrame({
            'time': df_raw[1].values,
            'CH1': df_raw[4].values,
            'CH2': df_raw[5].values
        }).dropna().iloc[5:] # Remove cabeçalhos
        
        return df, [ch_names[4], ch_names[5]]
    except Exception as e:
        st.error(f"Erro no processamento do arquivo: {e}")
        return None, None

# ==============================================================================
# 5. CONSTRUÇÃO DA INTERFACE DO USUÁRIO (UI)
# ==============================================================================

# 5.1 Barra Superior (Header)
col_h1, col_h2 = st.columns([4, 1])
with col_h2:
    selected_lang = st.selectbox("Language", list(LANGS.keys()), label_visibility="collapsed")
    tr = LANGS[selected_lang]
with col_h1:
    st.title(tr["title"])
    st.caption(tr["header_desc"])

# 5.2 Barra Lateral (Sidebar)
st.sidebar.markdown(f"### ⚙️ {tr['sidebar_title']}")
uploaded_file = st.sidebar.file_uploader(tr["upload_label"], type=["slk", "csv"])
st.sidebar.divider()
st.sidebar.info(tr["footer_msg"])

# 5.3 Lógica Principal de Processamento
if uploaded_file:
    df_emg, labels = parse_sylk(uploaded_file)
    
    if df_emg is not None:
        fs = 2000
        onsets_results = {} # Dicionário para guardar inícios de contração
        all_metrics_list = [] # Lista para exportação CSV
        
        # Grid de Canais (Lado a Lado)
        ui_cols = st.columns(2)

        for i, (ch_col, name) in enumerate(zip(['CH1', 'CH2'], labels)):
            with ui_cols[i]:
                st.markdown(f"### ⚡ {name}")
                
                # DSP
                signal_filtered = butter_bandpass_filter(df_emg[ch_col].values)
                rms_signal = calculate_rms(signal_filtered)
                
                # Plotagem com Plotly (Tema Escuro Hub Academica)
                fig = go.Figure(go.Scatter(
                    x=df_emg['time'], y=rms_signal,
                    line=dict(color=WHITE, width=1.5),
                    name="RMS Signal"
                ))
                
                fig.update_layout(
                    height=450,
                    margin=dict(l=10, r=10, t=20, b=10),
                    dragmode='select',
                    selectdirection='h',
                    plot_bgcolor=HUB_NAVY,
                    paper_bgcolor=HUB_NAVY,
                    newshape=dict(line=dict(color=HUB_BLUE, width=3), fillcolor=HUB_BLUE, opacity=0.4),
                    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color=WHITE), title="Tempo (s)"),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color=WHITE), title="Amplitude (µV)")
                )
                
                # Captura de Seleção
                selected_data = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key=f"plot_{ch_col}_{selected_lang}")

                # Se houver seleção no gráfico
                if selected_data and "selection" in selected_data and "box" in selected_data["selection"] and len(selected_data["selection"]["box"]) > 0:
                    t_start = selected_data["selection"]["box"][0]["x"][0]
                    t_end = selected_data["selection"]["box"][0]["x"][1]
                    
                    # Filtra dados da seleção
                    mask = (df_emg['time'] >= t_start) & (df_emg['time'] <= t_end)
                    data_sel_time = df_emg['time'][mask].values
                    data_sel_rms = rms_signal[mask]
                    
                    if len(data_sel_rms) > 20:
                        # Cálculo de Baseline (Primeiros 400ms do sinal original como referência)
                        baseline = rms_signal[:400]
                        threshold = np.mean(baseline) + (3 * np.std(baseline))
                        
                        # Detecção de Onset (Primeiros 40ms acima do threshold)
                        onset_idx = next((j for j in range(len(data_sel_rms)-80) if np.all(data_sel_rms[j:j+80] >= threshold)), None)
                        
                        # Métricas Adicionais
                        v_max = np.max(data_sel_rms)
                        v_mean = np.mean(data_sel_rms)
                        v_area = simpson(data_sel_rms, dx=1/fs)
                        v_onset = data_sel_time[onset_idx] if onset_idx is not None else None
                        
                        # Salva para comparação final
                        if v_onset: onsets_results[i] = v_onset
                        
                        all_metrics_list.append({
                            "Channel": name, "Onset": v_onset, "Peak": v_max, "Mean": v_mean, "Area": v_area
                        })

                        # Renderização do Card de Relatório
                        st.markdown(f"""
                        <div class="report-card">
                            <h4 style="margin-top:0;">{tr['rep']}</h4>
                            <div class="data-line"><span class="label-text">{tr['interval']}</span><span class="value-text">{t_start:.3f}s - {t_end:.3f}s</span></div>
                            <div class="data-line"><span class="label-text">{tr['duration']}</span><span class="value-text">{t_end-t_start:.3f} s</span></div>
                            <div class="data-line"><span class="label-text">{tr['ons']}</span><span class="value-text">{f"{v_onset:.3f} s" if v_onset else "N/D"}</span></div>
                            <div class="data-line"><span class="label-text">{tr['peak']}</span><span class="value-text">{v_max:.2f} µV</span></div>
                            <div class="data-line"><span class="label-text">{tr['mean_rms']}</span><span class="value-text">{v_mean:.2f} µV</span></div>
                            <div class="data-line"><span class="label-text">{tr['area']}</span><span class="value-text">{v_area:.4f} µV.s</span></div>
                            <div style="font-size:0.75em; color:gray; text-align:right; margin-top:10px;">{tr['thresh']}: {threshold:.4f} µV</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info(tr["info"])

        # ==========================================================
        # 6. SEÇÃO FINAL DE ANÁLISE COMPARATIVA E EXPORTAÇÃO
        # ==========================================================
        if len(all_metrics_list) > 0:
            st.divider()
            footer_col1, footer_col2 = st.columns([2, 1])
            
            with footer_col1:
                # Se ambos os canais tiverem Onset detectado, calcula o Delay
                if len(onsets_results) == 2:
                    delay_ms = abs(onsets_results[0] - onsets_results[1]) * 1000
                    st.success(f"### ⏱️ {tr['sync']}: **{delay_ms:.2f} ms**")
                elif len(all_metrics_list) == 2:
                    st.warning("⚠️ Onset não detectado em um dos canais para cálculo de Delay.")
            
            with footer_col2:
                # Gerador de CSV para download
                final_df = pd.DataFrame(all_metrics_list)
                csv_buffer = io.StringIO()
                final_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label=tr["dl_btn"],
                    data=csv_buffer.getvalue(),
                    file_name=f"EMG_Report_{int(time.time())}.csv",
                    mime="text/csv"
                )
else:
    # Estado inicial sem arquivo
    st.warning(f"### 📥 {tr['wait']}")
    st.image("https://img.icons8.com/clouds/200/statistics.png") # Ilustração amigável
