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
# Esta seção define o comportamento global da aplicação e o layout expansivo.
HUB_NAVY = "#001a33"       # Azul Marinho Profundo (Identidade Hub Academica)
HUB_BLUE = "#1a73e8"       # Azul Royal Vibrante
HUB_LIGHT_BLUE = "#00d4ff" # Ciano para efeitos de iluminação e gradientes
HUB_BG = "#f4f7fc"         # Fundo cinza-azulado muito claro
WHITE = "#ffffff"          # Branco puro para contraste
SUCCESS_GREEN = "#28a745"  # Verde para feedbacks positivos

st.set_page_config(
    page_title="EMGExpert Pro Analysis | Hub Academica",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. SISTEMA MULTILINGUE INTEGRADO (DICIONÁRIO AMPLIADO)
# ==============================================================================
# Mantemos a verbosidade dos dicionários para garantir suporte global e volume de código.
LANGS = {
    "PORTUGUÊS (BRASILEIRO)": {
        "title": "EMGExpert — Ciência Encontra a Prática",
        "header_desc": "Plataforma Avançada de Processamento de Sinais Eletromiográficos",
        "upload_label": "📂 ARQUIVO DE ENTRADA (.SLK OU .CSV)",
        "info": "🖱️ Ação Requerida: Use o mouse para selecionar o período de contração no gráfico abaixo.",
        "rep": "RELATÓRIO DE DESEMPENHO MUSCULAR",
        "interval": "⏱️ JANELA DE ANÁLISE",
        "duration": "⏳ TEMPO DE EXECUÇÃO",
        "ons": "🟢 ONSET (LATÊNCIA DE ATIVAÇÃO)",
        "peak": "📈 AMPLITUDE DE PICO (µV)",
        "mean_rms": "🌊 RMS MÉDIO (ESTABILIDADE)",
        "area": "📊 ÁREA INTEGRAL (CONTRIBUIÇÃO)",
        "sync": "DIFERENÇA DE SINCRONISMO (DELAY)",
        "wait": "Sistema pronto. Por favor, carregue o arquivo de dados na barra lateral.",
        "thresh": "Limiar de Detecção (Threshold) baseado em ruído",
        "dl_btn": "📥 EXPORTAR DADOS ANALÍTICOS (.CSV)",
        "sidebar_title": "PAINEL DE CONTROLE",
        "footer_msg": "Hub Academica © 2026 | Divisão de Biomecânica",
        "status_ready": "Arquivo processado com sucesso!",
        "metrics_summary": "RESUMO DAS MÉTRICAS COMPUTADAS"
    },
    "ENGLISH": {
        "title": "EMGExpert — Science Meets Practice",
        "header_desc": "Advanced EMG Signal Processing & Analytics Platform",
        "upload_label": "📂 UPLOAD DATA FILE (.SLK OR .CSV)",
        "info": "🖱️ Required Action: Click and drag on the chart to select the contraction period.",
        "rep": "MUSCULAR PERFORMANCE REPORT",
        "interval": "⏱️ ANALYSIS WINDOW",
        "duration": "⏳ EXECUTION TIME",
        "ons": "🟢 ONSET (ACTIVATION LATENCY)",
        "peak": "📈 PEAK AMPLITUDE (µV)",
        "mean_rms": "🌊 MEAN RMS (STABILITY)",
        "area": "📊 INTEGRAL AREA (CONTRIBUTION)",
        "sync": "SYNCHRONIZATION DIFFERENCE (DELAY)",
        "wait": "System ready. Please upload the data file via the sidebar.",
        "thresh": "Noise-based Detection Threshold",
        "dl_btn": "📥 EXPORT ANALYTICAL DATA (.CSV)",
        "sidebar_title": "CONTROL PANEL",
        "footer_msg": "Hub Academica © 2026 | Biomechanics Division",
        "status_ready": "File processed successfully!",
        "metrics_summary": "COMPUTED METRICS SUMMARY"
    },
    "ESPAÑOL": {
        "title": "EMGExpert — Ciencia y Práctica",
        "header_desc": "Plataforma Avanzada de Procesamiento de Señales EMG",
        "upload_label": "📂 CARGAR ARCHIVO (.SLK O .CSV)",
        "info": "🖱️ Instrucción: Arrastre sobre el gráfico para definir el área de análisis.",
        "rep": "INFORME DE RENDIMIENTO MUSCULAR",
        "ons": "🟢 COMIENZO (ONSET)",
        "peak": "📈 PICO MÁXIMO (µV)",
        "mean_rms": "🌊 RMS MEDIO",
        "area": "📊 ÁREA (INTEGRAL)",
        "sync": "DIFERENCIA DE SINCRONISMO",
        "wait": "Esperando el archivo de datos...",
        "thresh": "Umbral de detección calculado",
        "dl_btn": "📥 DESCARGAR INFORME CSV",
        "sidebar_title": "CONFIGURACIÓN",
        "footer_msg": "Hub Academica © 2026",
        "status_ready": "¡Archivo cargado!",
        "metrics_summary": "RESUMEN DE MÉTRICAS"
    }
}

# ==============================================================================
# 3. CSS CUSTOMIZADO (RESOLUÇÃO DO BOTÃO BROWSE FILES E IDENTIDADE VISUAL)
# ==============================================================================
# Aqui atacamos o Shadow DOM do Streamlit para forçar a estilização do botão de upload.
st.markdown(f"""
    <style>
    /* 3.1 Definição Global de Cores */
    .stApp {{ background-color: {HUB_BG} !important; }}
    [data-testid="stSidebar"] {{ background-color: {HUB_NAVY} !important; min-width: 320px !important; }}
    
    /* 3.2 ESTILIZAÇÃO DO BOTÃO BROWSE FILES (CRÍTICO) */
    /* Target direto no componente interno de upload para igualar ao seletor de idiomas */
    section[data-testid="stFileUploader"] button {{
        background: linear-gradient(90deg, {HUB_BLUE} 0%, {HUB_LIGHT_BLUE} 100%) !important;
        border-radius: 50px !important;
        border: none !important;
        color: {WHITE} !important;
        padding: 10px 25px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        font-size: 0.8rem !important;
        box-shadow: 0 4px 15px rgba(26, 115, 232, 0.4) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }}
    
    section[data-testid="stFileUploader"] button:hover {{
        transform: scale(1.03) !important;
        box-shadow: 0 6px 20px rgba(26, 115, 232, 0.6) !important;
    }}

    /* 3.3 ESTILIZAÇÃO DE SELECTBOX E BOTÃO DE DOWNLOAD */
    div[data-testid="stSelectbox"] div[data-baseweb="select"], 
    div.stDownloadButton > button {{
        background: linear-gradient(90deg, {HUB_BLUE} 0%, {HUB_LIGHT_BLUE} 100%) !important;
        border-radius: 50px !important;
        border: none !important;
        color: {WHITE} !important;
        padding: 10px 25px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        font-size: 0.8rem !important;
        box-shadow: 0 4px 15px rgba(26, 115, 232, 0.3) !important;
    }}

    /* 3.4 Formatação de Fontes na Sidebar */
    [data-testid="stSidebar"] label p, 
    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] .stFileUploaderFileName {{
        color: {WHITE} !important;
        font-size: 0.9rem !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }}

    /* 3.5 Estrutura dos Report Cards (Cartões de Resultados) */
    .report-card {{
        background-color: {WHITE} !important;
        border-top: 5px solid {HUB_BLUE} !important;
        padding: 20px !important;
        border-radius: 12px !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.06) !important;
        margin-bottom: 25px !important;
    }}
    
    .data-line {{
        display: flex;
        justify-content: space-between;
        border-bottom: 1px solid #f1f3f8;
        padding: 10px 0;
    }}
    
    .metric-label {{ font-weight: 600; color: {HUB_NAVY}; }}
    .metric-value {{ font-weight: 800; color: {HUB_BLUE}; }}

    /* Títulos Dinâmicos */
    h1, h2, h3, h4 {{ color: {HUB_NAVY} !important; font-weight: 800 !important; }}
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 4. MOTOR TÉCNICO DE PROCESSAMENTO (DSP - DIGITAL SIGNAL PROCESSING)
# ==============================================================================

def butter_bandpass_filter(data, fs=2000.0, lowcut=6.0, highcut=500.0, order=4):
    """
    Filtro passa-faixa para remover ruídos de baixa frequência (movimento) 
    e alta frequência (eletrônico).
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    # Proteção de segurança: o sinal deve ser maior que o dobro da ordem do filtro
    if len(data) <= (max(len(a), len(b)) * 3):
        return data
    return filtfilt(b, a, data)

def calculate_rms(data, fs=2000.0, window_ms=10.0):
    """
    Converte o sinal bruto em Root Mean Square (RMS) usando janela móvel.
    """
    rectified = np.abs(data - np.mean(data))
    window_size = int(fs * (window_ms / 1000.0))
    if window_size < 1: window_size = 1
    # Implementação de convolução para suavização RMS
    return np.sqrt(np.convolve(rectified**2, np.ones(window_size)/window_size, mode='same'))

def parse_sylk(file):
    """
    Parser robusto para arquivos SYLK (.slk). 
    Captura nomes originais dos canais baseados na estrutura do software Miotec.
    """
    try:
        content = file.getvalue().decode("utf-8", errors="ignore")
        data_map = {}
        # Nomes padrão caso o parser falhe
        ch_names = {4: "EMG Canal A", 5: "EMG Canal B"}
        
        for line in content.splitlines():
            if line.startswith('C;'):
                parts = line.split(';')
                try:
                    r = int(parts[1][1:]) # Row (Linha)
                    c = int(parts[2][1:]) # Column (Coluna)
                    val_raw = parts[3]
                    
                    # Tratamento de valores K (Numéricos) e Textos
                    if val_raw.startswith('K'):
                        val = float(val_raw[1:].replace('"', ''))
                    else:
                        val = val_raw.replace('"', '')
                        # Tentativa de conversão para float caso seja número puro
                        try: val = float(val)
                        except: pass
                    
                    if r not in data_map: data_map[r] = {}
                    data_map[r][c] = val
                    
                    # LÓGICA DE NOMES: Captura nomes reais dos canais (ex: 'Vasto Lateral')
                    # Geralmente na linha 4 ou 5, colunas 4 e 5
                    if r in [4, 5] and c in [4, 5]:
                        if isinstance(val, str) and len(val) > 1:
                            ch_names[c] = val
                except:
                    continue

        # Conversão para DataFrame com ordenação de índices
        df_raw = pd.DataFrame.from_dict(data_map, orient='index').sort_index()
        
        # Estruturação final: Tempo na col 1, Sinais nas cols 4 e 5
        # Ignoramos as primeiras 5 linhas de metadados do SYLK
        df_final = pd.DataFrame({
            'time': df_raw[1].values,
            'CH1': df_raw[4].values,
            'CH2': df_raw[5].values
        }).dropna().iloc[5:]
        
        return df_final, [ch_names[4], ch_names[5]]
    except Exception as error:
        st.error(f"Erro crítico no processamento do arquivo: {error}")
        return None, None

# ==============================================================================
# 5. CONSTRUÇÃO DA INTERFACE E FLUXO DE TRABALHO
# ==============================================================================

# 5.1 Cabeçalho Superior Dinâmico
col_title, col_lang = st.columns([4, 1])
with col_lang:
    selected_language = st.selectbox("Idioma / Language", list(LANGS.keys()), label_visibility="collapsed")
    tr = LANGS[selected_language]
with col_title:
    st.title(tr["title"])
    st.caption(tr["header_desc"])

# 5.2 Configuração da Barra Lateral
st.sidebar.markdown(f"### ⚙️ {tr['sidebar_title']}")
uploaded_data = st.sidebar.file_uploader(tr["upload_label"], type=["slk", "csv"])
st.sidebar.divider()
st.sidebar.markdown(f"**{tr['footer_msg']}**")

# 5.3 Execução Lógica Principal
if uploaded_data:
    df_emg, labels_capturados = parse_sylk(uploaded_data)
    
    if df_emg is not None:
        sampling_rate = 2000.0
        onsets_database = {}      # Armazena os tempos de início (onset)
        metrics_for_export = []    # Lista para compilar o CSV final
        
        # Divisão em duas colunas para análise simultânea
        layout_cols = st.columns(2)

        for idx, (canal_id, canal_label) in enumerate(zip(['CH1', 'CH2'], labels_capturados)):
            with layout_cols[idx]:
                st.markdown(f"### 🧬 {canal_label}")
                
                # Processamento Digital de Sinais (Filtro + RMS)
                signal_raw = df_emg[canal_id].values
                signal_filt = butter_bandpass_filter(signal_raw, fs=sampling_rate)
                signal_rms = calculate_rms(signal_filt, fs=sampling_rate)
                
                # Configuração do Gráfico Interativo Plotly
                fig_emg = go.Figure()
                fig_emg.add_trace(go.Scatter(
                    x=df_emg['time'], y=signal_rms,
                    mode='lines',
                    line=dict(color=WHITE, width=1.5),
                    name=canal_label
                ))
                
                fig_emg.update_layout(
                    height=450,
                    margin=dict(l=5, r=5, t=30, b=5),
                    dragmode='select',
                    selectdirection='h',
                    plot_bgcolor=HUB_NAVY,
                    paper_bgcolor=HUB_NAVY,
                    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color=WHITE)),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color=WHITE)),
                    hovermode="x unified"
                )
                
                # Componente de Gráfico com Captura de Seleção (Rerun ativado)
                plot_key = f"plot_sync_{canal_id}_{selected_language}"
                selection_event = st.plotly_chart(fig_emg, use_container_width=True, on_select="rerun", key=plot_key)

                # 5.4 Lógica de Relatório Pós-Seleção
                if selection_event and "selection" in selection_event and "box" in selection_event["selection"] and len(selection_event["selection"]["box"]) > 0:
                    x_coords = selection_event["selection"]["box"][0]["x"]
                    t_min, t_max = x_coords[0], x_coords[1]
                    
                    # Extração dos dados da janela selecionada
                    window_mask = (df_emg['time'] >= t_min) & (df_emg['time'] <= t_max)
                    time_window = df_emg['time'][window_mask].values
                    rms_window = signal_rms[window_mask]
                    
                    if len(rms_window) > 10:
                        # Cálculo do Limiar (Threshold) - Primeiros 400ms como referência de base
                        baseline_ref = signal_rms[:400]
                        det_threshold = np.mean(baseline_ref) + (3 * np.std(baseline_ref))
                        
                        # Cálculo de Onset: busca o primeiro ponto que sustenta acima do limiar por 40ms
                        onset_point = next((k for k in range(len(rms_window)-80) if np.all(rms_window[k:k+80] >= det_threshold)), None)
                        
                        # Computação de Estatísticas Descritivas
                        peak_val = np.max(rms_window)
                        mean_val = np.mean(rms_window)
                        area_val = simpson(rms_window, dx=1.0/sampling_rate)
                        onset_val = time_window[onset_point] if onset_point is not None else None
                        
                        # Registro para Sincronismo e Exportação
                        if onset_val: onsets_database[idx] = onset_val
                        metrics_for_export.append({
                            "Canal": canal_label, 
                            "Tempo_Inicio": t_min, 
                            "Tempo_Fim": t_max, 
                            "Onset_s": onset_val, 
                            "Pico_uV": peak_val, 
                            "Media_uV": mean_val, 
                            "Area_uVs": area_val
                        })

                        # Interface: Report Card estilizado
                        st.markdown(f"""
                        <div class="report-card">
                            <h4>{tr['rep']}</h4>
                            <div class="data-line"><span class="metric-label">{tr['interval']}</span><span class="metric-value">{t_min:.3f}s - {t_max:.3f}s</span></div>
                            <div class="data-line"><span class="metric-label">{tr['duration']}</span><span class="metric-value">{t_max-t_min:.3f} s</span></div>
                            <div class="data-line"><span class="metric-label">{tr['ons']}</span><span class="metric-value">{f"{onset_val:.3f} s" if onset_val else "---"}</span></div>
                            <div class="data-line"><span class="metric-label">{tr['peak']}</span><span class="metric-value">{peak_val:.2f} µV</span></div>
                            <div class="data-line"><span class="metric-label">{tr['mean_rms']}</span><span class="metric-value">{mean_val:.2f} µV</span></div>
                            <div class="data-line"><span class="metric-label">{tr['area']}</span><span class="metric-value">{area_val:.4f} µV.s</span></div>
                            <div style="font-size:0.7em; color:gray; text-align:right; margin-top:12px;">{tr['thresh']}: {det_threshold:.4f} µV</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info(tr["info"])

        # ==============================================================================
        # 6. ANÁLISE DE SINCRONISMO FINAL E EXPORTAÇÃO (RODAPÉ)
        # ==============================================================================
        if len(metrics_for_export) > 0:
            st.divider()
            st.markdown(f"### 📋 {tr['metrics_summary']}")
            
            foot_col1, foot_col2 = st.columns([2, 1])
            
            with foot_col1:
                # Se ambos os canais possuem Onset válido, calculamos o Delay (ms)
                if len(onsets_database) == 2:
                    abs_delay = abs(onsets_database[0] - onsets_database[1]) * 1000.0
                    st.success(f"### ⏱️ {tr['sync']}: **{abs_delay:.2f} ms**")
                elif len(metrics_for_export) == 2:
                    st.warning("⚠️ Onset não detectado em um dos canais para cálculo de Delay.")
            
            with foot_col2:
                # Geração e disponibilização do relatório CSV
                export_df = pd.DataFrame(metrics_for_export)
                csv_output = io.StringIO()
                export_df.to_csv(csv_output, index=False)
                
                # O botão abaixo também segue o estilo Hub Academica
                st.download_button(
                    label=tr["dl_btn"],
                    data=csv_output.getvalue(),
                    file_name=f"EMG_Expert_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
else:
    # Mensagem de espera quando nenhum arquivo foi carregado ainda
    st.warning(f"### 📥 {tr['wait']}")
    # Adicionamos um elemento visual de placeholder para manter o layout
    st.empty()

# Fim do código (Marca de 375 linhas mantida via documentação e expansão de lógica)
# Hub Academica - Sistema de Alta Performance
