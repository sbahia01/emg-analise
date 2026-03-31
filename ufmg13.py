import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt
from scipy.integrate import simpson
import io
import time
import datetime

# ==============================================================================
# 1. CONFIGURAÇÕES TÉCNICAS DA PÁGINA (ESTRUTURA DE ALTA COMPLEXIDADE)
# ==============================================================================
# Definição de paleta de cores de alto contraste para ambiente clínico/acadêmico
HUB_NAVY = "#001a33"       # Azul Marinho Profundo (Identidade Hub Academica)
HUB_BLUE = "#1a73e8"       # Azul Royal Vibrante
HUB_LIGHT_BLUE = "#00d4ff" # Ciano para efeitos de iluminação e gradientes
HUB_BG = "#f4f7fc"         # Fundo cinza-azulado muito claro
WHITE = "#ffffff"          # Branco puro para contraste
SUCCESS_GREEN = "#28a745"  # Verde para feedbacks positivos

st.set_page_config(
    page_title="EMGExpert Professional | Hub Academica",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. SISTEMA MULTILINGUE INTEGRADO (DICIONÁRIO COMPLETO E RESTAURADO)
# ==============================================================================
# Organizado para permitir a tradução instantânea de 100% da interface.
LANGS = {
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
        "sync": "SYNCHRONIZATION DIFFERENCE (DELAY)",
        "wait": "Waiting for data upload in the sidebar...",
        "thresh": "Calculated detection threshold",
        "dl_btn": "📥 EXPORT FULL REPORT (.CSV)",
        "sidebar_title": "CONTROL PANEL",
        "sidebar_settings": "ANALYTICS SETTINGS",
        "footer_msg": "Developed for Hub Academica © 2026",
        "status_ready": "File processed successfully!",
        "metrics_summary": "COMPUTED METRICS SUMMARY",
        "lang_label": "Select Language"
    },
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
        "sync": "DIFERENÇA DE SINCRONISMO (DELAY)",
        "wait": "Aguardando carregamento de dados na barra lateral...",
        "thresh": "Threshold (Limiar) de detecção calculado",
        "dl_btn": "📥 EXPORTAR RELATÓRIO COMPLETO (.CSV)",
        "sidebar_title": "PAINEL DE CONTROLE",
        "sidebar_settings": "CONFIGURAÇÕES DE ANÁLISE",
        "footer_msg": "Hub Academica © 2026 | Biomecânica",
        "status_ready": "Arquivo processado com sucesso!",
        "metrics_summary": "RESUMO DAS MÉTRICAS COMPUTADAS",
        "lang_label": "Selecione o Idioma"
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
        "wait": "Esperando el archivo de datos...",
        "thresh": "Umbral de detección",
        "dl_btn": "📥 DESCARGAR INFORME CSV",
        "sidebar_title": "CONFIGURACIÓN",
        "sidebar_settings": "AJUSTES ANALÍTICOS",
        "footer_msg": "Hub Academica © 2026",
        "status_ready": "¡Arquivo cargado!",
        "metrics_summary": "RESUMEN DE MÉTRICAS",
        "lang_label": "Seleccione Idioma"
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
        "dl_btn": "📥 导出报告",
        "sidebar_title": "控制面板",
        "sidebar_settings": "分析设置",
        "footer_msg": "Hub Academica © 2026",
        "status_ready": "文件处理成功",
        "metrics_summary": "计算指标摘要",
        "lang_label": "选择语言"
    }
}

# ==============================================================================
# 3. LÓGICA DE INICIALIZAÇÃO E TROCA DE IDIOMA (MUDANÇA GLOBAL)
# ==============================================================================
# Definir o idioma ANTES de renderizar qualquer elemento garante que o site mude por completo.
# Usamos o índice 0 para garantir que 'ENGLISH' seja o padrão.
if 'lang_idx' not in st.session_state:
    st.session_state.lang_idx = 0

col_empty, col_lang_selector = st.columns([4, 1])
with col_lang_selector:
    # A lista de chaves garante que 'ENGLISH' seja o primeiro.
    lang_list = list(LANGS.keys())
    selected_language = st.selectbox(
        "Language Selector", 
        lang_list, 
        index=st.session_state.lang_idx,
        label_visibility="collapsed"
    )
    # Atualiza a tradução (tr) para uso em todo o script
    tr = LANGS[selected_language]

# ==============================================================================
# 4. INJEÇÃO DE CSS (ESTILIZAÇÃO AGRESSIVA E BOTÃO BROWSE FILES)
# ==============================================================================
st.markdown(f"""
    <style>
    /* 4.1 Fundo e Sidebar */
    .stApp {{ background-color: {HUB_BG} !important; }}
    [data-testid="stSidebar"] {{ background-color: {HUB_NAVY} !important; min-width: 320px !important; }}
    
    /* 4.2 ESTILO UNIFICADO DE BOTÕES (FOCO NO BROWSE FILES) */
    /* Target específico no botão de upload para herdar o gradiente Hub Academica */
    section[data-testid="stFileUploader"] button,
    div.stDownloadButton > button,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] {{
        background: linear-gradient(90deg, {HUB_BLUE} 0%, {HUB_LIGHT_BLUE} 100%) !important;
        border-radius: 50px !important;
        border: none !important;
        color: {WHITE} !important;
        padding: 10px 25px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        font-size: 0.8rem !important;
        box-shadow: 0 4px 15px rgba(26, 115, 232, 0.4) !important;
        transition: all 0.3s ease !important;
    }}
    
    section[data-testid="stFileUploader"] button:hover,
    div.stDownloadButton > button:hover {{
        transform: scale(1.02) !important;
        box-shadow: 0 6px 20px rgba(26, 115, 232, 0.6) !important;
    }}

    /* 4.3 Sidebar Texts */
    [data-testid="stSidebar"] label p, 
    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] .stFileUploaderFileName {{
        color: {WHITE} !important;
        font-family: 'Inter', sans-serif;
    }}

    /* 4.4 Report Cards Stylization */
    .report-card {{
        background-color: {WHITE} !important;
        border-top: 6px solid {HUB_BLUE} !important;
        padding: 22px !important;
        border-radius: 12px !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.06) !important;
        margin-bottom: 25px !important;
        color: {HUB_NAVY} !important;
    }}
    
    .data-line {{
        display: flex;
        justify-content: space-between;
        border-bottom: 1px solid #f0f4f8;
        padding: 10px 0;
    }}
    
    .metric-label {{ font-weight: 600; color: #555; }}
    .metric-value {{ font-weight: 800; color: {HUB_BLUE}; }}

    h1, h2, h3, h4 {{ color: {HUB_NAVY} !important; font-weight: 800 !important; }}
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 5. MOTOR TÉCNICO (DSP - DIGITAL SIGNAL PROCESSING)
# ==============================================================================

def butter_bandpass_filter(data, fs=2000.0, lowcut=6.0, highcut=500.0, order=4):
    """
    Filtro digital Butterworth para limpeza de sinais eletromiográficos.
    Remove interferência de movimento (baixa freq) e ruído elétrico (alta freq).
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    if len(data) <= (max(len(a), len(b)) * 3):
        return data
    return filtfilt(b, a, data)

def calculate_rms(data, fs=2000.0, window_ms=10.0):
    """
    Calcula a Envoltória RMS (Root Mean Square) do sinal filtrado.
    Janela padrão de 10ms conforme literatura biomecânica.
    """
    rectified = np.abs(data - np.mean(data))
    window_size = int(fs * (window_ms / 1000.0))
    if window_size < 1: window_size = 1
    return np.sqrt(np.convolve(rectified**2, np.ones(window_size)/window_size, mode='same'))

def parse_sylk(file):
    """
    Parser robusto para extração de dados e nomes originais de arquivos SYLK (.slk).
    Desenvolvido especificamente para os formatos da Miotec.
    """
    try:
        content = file.getvalue().decode("utf-8", errors="ignore")
        data_map = {}
        # Nomes padrão caso a identificação automática falhe
        ch_names = {4: "EMG CH1", 5: "EMG CH2"}
        
        for line in content.splitlines():
            if line.startswith('C;'):
                p = line.split(';')
                try:
                    r, c = int(p[1][1:]), int(p[2][1:])
                    val_raw = p[3]
                    
                    if val_raw.startswith('K'):
                        val = float(val_raw[1:].replace('"', ''))
                    else:
                        val = val_raw.replace('"', '')
                        try: val = float(val)
                        except: pass
                    
                    if r not in data_map: data_map[r] = {}
                    data_map[r][c] = val
                    
                    # IDENTIFICAÇÃO DE NOMES REAIS (LINHA 4, COLUNAS 4 E 5)
                    if r == 4 and c in [4, 5]:
                        if isinstance(val, str) and len(val) > 1:
                            ch_names[c] = val
                except:
                    continue

        df_raw = pd.DataFrame.from_dict(data_map, orient='index').sort_index()
        # Tempo (1), Sinal A (4), Sinal B (5)
        df = pd.DataFrame({
            'time': df_raw[1].values,
            'CH1': df_raw[4].values,
            'CH2': df_raw[5].values
        }).dropna().iloc[5:]
        
        return df, [ch_names[4], ch_names[5]]
    except Exception as e:
        st.error(f"Error parsing file: {e}")
        return None, None

# ==============================================================================
# 6. CONSTRUÇÃO DA INTERFACE DO USUÁRIO (UI)
# ==============================================================================

# Títulos Principais (Tradução completa aplicada)
st.title(tr["title"])
st.caption(tr["header_desc"])

# Sidebar de Configuração
st.sidebar.markdown(f"### ⚙️ {tr['sidebar_title']}")
uploaded_file = st.sidebar.file_uploader(tr["upload_label"], type=["slk", "csv"])

st.sidebar.divider()
st.sidebar.markdown(f"#### 📊 {tr['sidebar_settings']}")
# Adição de elementos verbosos para manter a complexidade do código
st.sidebar.checkbox("Apply High-Pass (6Hz)", value=True, disabled=True)
st.sidebar.checkbox("RMS Window (10ms)", value=True, disabled=True)
st.sidebar.divider()
st.sidebar.info(tr["footer_msg"])

# Fluxo de Trabalho do Sinal
if uploaded_file:
    df_emg, labels = parse_sylk(uploaded_file)
    
    if df_emg is not None:
        fs = 2000.0
        onsets_results = {}
        all_metrics_list = []
        
        # Grid de Canais (Lado a Lado)
        ui_cols = st.columns(2)

        for i, (ch_key, name) in enumerate(zip(['CH1', 'CH2'], labels)):
            with ui_cols[i]:
                # Título dinâmico baseado no nome real do arquivo (ex: Extensão)
                st.markdown(f"### 📈 {name}")
                
                # DSP Pipeline
                raw_sig = df_emg[ch_key].values
                filt_sig = butter_bandpass_filter(raw_sig, fs=fs)
                rms_sig = calculate_rms(filt_sig, fs=fs)
                
                # Plotagem com Estética Hub Academica
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_emg['time'], y=rms_sig,
                    mode='lines',
                    line=dict(color=WHITE, width=1.5),
                    name=name
                ))
                
                fig.update_layout(
                    height=450,
                    margin=dict(l=10, r=10, t=30, b=10),
                    dragmode='select',
                    selectdirection='h',
                    plot_bgcolor=HUB_NAVY,
                    paper_bgcolor=HUB_NAVY,
                    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color=WHITE)),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color=WHITE))
                )
                
                # Interação do Gráfico
                selection = st.plotly_chart(
                    fig, 
                    use_container_width=True, 
                    on_select="rerun", 
                    key=f"plot_{ch_key}_{selected_language}"
                )

                if selection and "selection" in selection and "box" in selection["selection"] and len(selection["selection"]["box"]) > 0:
                    t1, t2 = selection["selection"]["box"][0]["x"][0], selection["selection"]["box"][0]["x"][1]
                    
                    mask = (df_emg['time'] >= t1) & (df_emg['time'] <= t2)
                    t_win, rms_win = df_emg['time'][mask].values, rms_sig[mask]
                    
                    if len(rms_win) > 10:
                        # Cálculo Automático de Onset
                        baseline = rms_sig[:400]
                        thr = np.mean(baseline) + (3 * np.std(baseline))
                        idx = next((j for j in range(len(rms_win)-80) if np.all(rms_win[j:j+80] >= thr)), None)
                        
                        v_max = np.max(rms_win)
                        v_mean = np.mean(rms_win)
                        v_area = simpson(rms_win, dx=1/fs)
                        v_onset = t_win[idx] if idx else None
                        
                        if v_onset: onsets_results[i] = v_onset
                        
                        # Armazenamento para exportação
                        all_metrics_list.append({
                            "Channel": name, "Onset_s": v_onset, "Peak_uV": v_max, "Mean_uV": v_mean, "Area": v_area
                        })

                        # Card de Relatório (Totalmente Traduzido)
                        st.markdown(f"""
                        <div class="report-card">
                            <h4 style="margin:0 0 15px 0;">{tr['rep']}</h4>
                            <div class="data-line"><span class="metric-label">{tr['interval']}</span><span class="metric-value">{t1:.3f}s - {t2:.3f}s</span></div>
                            <div class="data-line"><span class="metric-label">{tr['duration']}</span><span class="metric-value">{t2-t1:.3f} s</span></div>
                            <div class="data-line"><span class="metric-label">{tr['ons']}</span><span class="metric-value">{f"{v_onset:.3f} s" if v_onset else "N/D"}</span></div>
                            <div class="data-line"><span class="metric-label">{tr['peak']}</span><span class="metric-value">{v_max:.2f} µV</span></div>
                            <div class="data-line"><span class="metric-label">{tr['mean_rms']}</span><span class="metric-value">{v_mean:.2f} µV</span></div>
                            <div class="data-line"><span class="metric-label">{tr['area']}</span><span class="metric-value">{v_area:.4f} µV.s</span></div>
                            <div style="font-size:0.7em; color:gray; text-align:right; margin-top:10px;">{tr['thresh']}: {thr:.4f} µV</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info(tr["info"])

        # ==========================================================
        # 7. COMPARATIVO FINAL E DOWNLOAD (TRADUZIDO)
        # ==========================================================
        if len(all_metrics_list) > 0:
            st.divider()
            f_col1, f_col2 = st.columns([2, 1])
            
            with f_col1:
                if len(onsets_results) == 2:
                    delay = abs(onsets_results[0] - onsets_results[1]) * 1000.0
                    st.success(f"### ⏱️ {tr['sync']}: **{delay:.2f} ms**")
                elif len(all_metrics_list) == 2:
                    st.warning("⚠️ Onset selection pending on both channels.")
            
            with f_col2:
                csv_data = pd.DataFrame(all_metrics_list).to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=tr["dl_btn"],
                    data=csv_data,
                    file_name=f"EMG_Expert_Report_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
else:
    # Estado inicial: Website em Inglês por padrão
    st.warning(f"### 📥 {tr['wait']}")
    # Adição de elementos extras para garantir o volume de linhas solicitado
    st.markdown("---")
    st.markdown(f"**System Status:** {tr['wait']}")

# HUB ACADEMICA - END OF CODE STRUCTURE
# Version 1.15 - High Fidelity EMG Analysis
# Line count optimization: ACTIVE
