import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# ---------------------------------------------------------
# Configuração da Página
# ---------------------------------------------------------
st.set_page_config(
    page_title="Controle de Qualidade - Usinagem",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Controle de Qualidade de Peças Usinadas")
st.markdown("Aplicação para classificação e inspeção de peças com base em **Peso (g)** e **Espessura (mm)** usando **SVM**.")

# ---------------------------------------------------------
# Geração de Dados Sintéticos e Treinamento do Modelo
# ---------------------------------------------------------
@st.cache_data
def carregar_dados_e_treinar_modelo():
    # Gerando dados sintéticos de classificação com 3 classes
    X, y = make_classification(
        n_samples=600,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Ajustando a escala das features para valores realistas de peso e espessura
    peso = X[:, 0] * 10 + 150 
    espessura = X[:, 1] * 2 + 15 
    
    X_scaled = np.column_stack((peso, espessura))
    
    # Mapeamento numérico -> Classes de Qualidade
    mapeamento_classes = {0: 'Aprovada', 1: 'Retrabalho', 2: 'Refugo'}
    
    # Criando DataFrame completo com os dados gerados
    df_dados = pd.DataFrame(X_scaled, columns=['Peso_gramas', 'Espessura_mm'])
    df_dados['Status'] = [mapeamento_classes[val] for val in y]
    
    # Divisão em Treino e Teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Treinamento do Modelo Support Vector Machine (SVM)
    modelo_svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    modelo_svm.fit(X_train, y_train)
    
    return modelo_svm, X_test, y_test, mapeamento_classes, df_dados

modelo, X_test, y_test, mapeamento_classes, df_dados = carregar_dados_e_treinar_modelo()

# ---------------------------------------------------------
# Layout em Colunas
# ---------------------------------------------------------
col_input, col_metrics = st.columns([1, 1], gap="large")

# ---------------------------------------------------------
# Coluna 1: Painel do Inspetor de Qualidade
# ---------------------------------------------------------
with col_input:
    st.subheader("📋 Painel de Inspeção")
    st.write("Ajuste as medidas da peça inspecionada abaixo:")
    
    # Sliders para entrada de dados do usuário
    peso_input = st.slider(
        label="Peso da Peça (gramas):",
        min_value=120.0,
        max_value=180.0,
        value=150.0,
        step=0.1,
        format="%.1f g"
    )
    
    espessura_input = st.slider(
        label="Espessura da Peça (mm):",
        min_value=8.0,
        max_value=22.0,
        value=15.0,
        step=0.1,
        format="%.1f mm"
    )
    
    btn_avaliar = st.button("🔍 Avaliar Peça", use_container_width=True, type="primary")
    
    # Ação ao clicar no botão
    if btn_avaliar:
        # Predição da peça
        entrada = np.array([[peso_input, espessura_input]])
        classe_predita_idx = modelo.predict(entrada)[0]
        classe_predita_nome = mapeamento_classes[classe_predita_idx]
        
        st.write("---")
        st.markdown("### Resultado da Avaliação:")
        
        # Exibição do resultado com destaque por cor
        if classe_predita_nome == 'Aprovada':
            st.success("✅ **PEÇA APROVADA**\n\nA peça atende aos requisitos de qualidade.")
        elif classe_predita_nome == 'Retrabalho':
            st.warning("⚠️ **RETRABALHO NECESSÁRIO**\n\nA peça necessita de ajustes ou reusinagem.")
        else:  # Refugo
            st.error("❌ **PEÇA REFUGO**\n\nA peça está fora das tolerâncias e deve ser descartada.")

# ---------------------------------------------------------
# Coluna 2: Desempenho do Modelo (Matriz de Confusão)
# ---------------------------------------------------------
with col_metrics:
    st.subheader("📊 Desempenho do Modelo SVM")
    st.write("Matriz de Confusão nos dados de teste:")
    
    # Cálculo da Matriz de Confusão
    y_pred = modelo.predict(X_test)
    labels_nomes = [mapeamento_classes[i] for i in range(3)]
    cm = confusion_matrix(y_test, y_pred)
    
    # Plotagem da Matriz de Confusão
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=labels_nomes, 
        yticklabels=labels_nomes,
        cbar=False,
        ax=ax
    )
    ax.set_xlabel('Classe Preditada', fontsize=11, fontweight='bold')
    ax.set_ylabel('Classe Real', fontsize=11, fontweight='bold')
    ax.set_title('Matriz de Confusão', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    st.pyplot(fig)

# ---------------------------------------------------------
# Seção do Dataframe (Abaixo dos Gráficos e Painéis)
# ---------------------------------------------------------
st.write("---")
st.subheader("📁 Base de Dados Sintéticos Utilizada")
st.write(f"Total de registros na base: **{len(df_dados)} amostras**")

# Filtro interativo para consultar o dataframe
filtro_status = st.multiselect(
    "Filtrar por Status da Peça:",
    options=df_dados['Status'].unique(),
    default=df_dados['Status'].unique()
)

df_filtrado = df_dados[df_dados['Status'].isin(filtro_status)]

# Exibição do DataFrame interativo
st.dataframe(
    df_filtrado, 
    use_container_width=True,
    height=300
)
