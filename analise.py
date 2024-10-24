# Importações necessárias
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import seaborn as sns

# Bibliotecas adicionais
import streamlit as st
import plotly.express as px
from collections import Counter
from wordcloud import WordCloud

# Configurar a página do Streamlit (removendo o layout wide)
st.set_page_config(page_title="Dashboard de Agendamentos Médicos")

# Certifique-se de baixar os recursos do NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('rslp')

def main():
    st.title("Análise de Agendamentos Médicos e de Enfermagem")

    # Upload do arquivo Excel
    file_upload = st.file_uploader("Selecione o arquivo Excel", type=["xlsx"])

    if file_upload is not None:
        # Ler a aba "Observações - APS" do arquivo Excel
        df = pd.read_excel(file_upload, sheet_name='Observações - APS')

        # Processar os dados conforme as especificações
        df_processed = processar_dados(df)

        # Filtro por período
        st.sidebar.subheader("Filtro por Período")
        min_date = df_processed['Data'].min().date()
        max_date = df_processed['Data'].max().date()
        start_date = st.sidebar.date_input('Data Inicial', min_value=min_date, max_value=max_date, value=min_date)
        end_date = st.sidebar.date_input('Data Final', min_value=min_date, max_value=max_date, value=max_date)
        if start_date > end_date:
            st.sidebar.error('Data Inicial não pode ser posterior à Data Final')
            return
        else:
            df_filtered = df_processed[(df_processed['Data'].dt.date >= start_date) & (df_processed['Data'].dt.date <= end_date)]

        # Análise exploratória de dados
        analise_exploratoria(df_filtered)

        # Exibir tabela de dados
        st.subheader("Dados Processados")
        st.write(df_filtered.head())
        st.markdown("_Descrição: Esta tabela mostra as primeiras linhas dos dados após o processamento, incluindo as novas colunas e ajustes realizados._", unsafe_allow_html=True)

    else:
        st.warning("Por favor, carregue um arquivo Excel.")

def processar_dados(df):
    st.info("Processando os dados...")

    # Remover espaços em branco e converter para string
    df['Status agenda'] = df['Status agenda'].astype(str).str.strip()
    df['Classificação'] = df['Classificação'].astype(str).str.strip()

    # Manter todas as linhas para análise completa

    # Listas de agendas dos médicos e enfermeiras
    agendas_medicos = ['Dra. Sheila', 'Dra. Leticia', 'Dra. Ana Paula']
    agendas_enfermeiras = ['Enfermeira Nelissa (Personal)', 'Enfermeira Tayara (Personal)', 'Enfermeira Mariana (Personal)']

    # Mapeamento de agendas para classificações das enfermeiras
    mapeamento_enfermeiras = {
        'Enfermeira Nelissa (Personal)': 'Enf. Nelissa',
        'Enfermeira Tayara (Personal)': 'Enf. Tayara',
        'Enfermeira Mariana (Personal)': 'Enf. Mariana'
    }

    # Criar uma nova coluna para identificar se é médico ou enfermeira
    def identificar_profissional(agenda):
        if agenda in agendas_medicos:
            return 'Médico'
        elif agenda in agendas_enfermeiras:
            return 'Enfermeira'
        else:
            return 'Outro'

    df['Tipo Profissional'] = df['Agenda'].apply(identificar_profissional)

    # Ajustar a coluna 'Classificação' para as enfermeiras
    def ajustar_classificacao(row):
        if row['Tipo Profissional'] == 'Enfermeira':
            return mapeamento_enfermeiras.get(row['Agenda'], row['Classificação'])
        elif row['Tipo Profissional'] == 'Médico':
            if row['Classificação'] in ['Demanda espontânea', 'Programada', 'Urgência']:
                return row['Classificação']
            else:
                return 'Outro'
        else:
            return 'Outro'

    df['Classificação Ajustada'] = df.apply(ajustar_classificacao, axis=1)

    # Filtrar apenas as classificações desejadas
    valid_classificacoes_medicos = ['Demanda espontânea', 'Programada', 'Urgência']
    valid_classificacoes_enfermeiras = ['Enf. Mariana', 'Enf. Nelissa', 'Enf. Tayara']
    df = df[df['Classificação Ajustada'].isin(valid_classificacoes_medicos + valid_classificacoes_enfermeiras)]

    # Preencher valores ausentes na coluna 'Observação' com string vazia
    df['Observação'] = df['Observação'].fillna('')

    # Converter coluna 'Data' para datetime
    df['Data'] = pd.to_datetime(df['Data'])

    # Criar coluna 'Dia da Semana' com números de 0 (segunda-feira) a 6 (domingo)
    df['Dia da Semana Num'] = df['Data'].dt.dayofweek
    dias_semana_dict = {
        0: 'Segunda-feira',
        1: 'Terça-feira',
        2: 'Quarta-feira',
        3: 'Quinta-feira',
        4: 'Sexta-feira',
        5: 'Sábado',
        6: 'Domingo'
    }
    df['Dia da Semana'] = df['Dia da Semana Num'].map(dias_semana_dict)

    # Pré-processamento do texto
    df['Observação Processada'] = df['Observação'].apply(preprocessar_texto)

    # Criar coluna 'Hora'
    df['Hora'] = df['Data'].dt.hour

    # Definir status ocupados e livres
    status_ocupado = ['Executada', 'Atendido', 'Falta não justificada', 'Falta justificada', 'Aguardando', 'Em Consulta']
    status_livre = ['Livre', 'Normal']
    status_bloqueada = ['Bloqueada']

    # Criar coluna 'Ocupado'
    df['Ocupado'] = np.where(df['Status agenda'].isin(status_ocupado), 'Ocupado',
                      np.where(df['Status agenda'].isin(status_livre), 'Livre', 'Bloqueado'))

    # Dividir os dados em treinamento e teste (apenas para as vagas ocupadas)
    df_ml = df[(df['Tipo Profissional'] == 'Médico') & (df['Ocupado'] == 'Ocupado')]
    df_ml = df_ml[df_ml['Classificação Ajustada'].isin(['Demanda espontânea', 'Programada', 'Urgência'])]

    X = df_ml['Observação Processada']
    y = df_ml['Classificação Ajustada']

    if not X.empty:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Criar um pipeline de processamento de texto e modelo de classificação
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', MultinomialNB()),
        ])

        # Treinar o modelo
        pipeline.fit(X_train, y_train)

        # Fazer previsões no conjunto de teste
        y_pred = pipeline.predict(X_test)

        # Avaliar o modelo
        st.subheader("Relatório de Classificação")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

        st.markdown("_Descrição: Este relatório mostra as métricas de desempenho do modelo de aprendizado de máquina na classificação das consultas médicas._", unsafe_allow_html=True)

        # Matriz de confusão
        st.subheader("Matriz de Confusão")
        cm = confusion_matrix(y_test, y_pred, labels=['Urgência', 'Demanda espontânea', 'Programada'])
        fig_cm = plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Urgência', 'Demanda espontânea', 'Programada'], yticklabels=['Urgência', 'Demanda espontânea', 'Programada'], cbar=False)
        plt.ylabel('Verdadeiro')
        plt.xlabel('Previsto')
        plt.title('Matriz de Confusão')
        st.pyplot(fig_cm)

        st.markdown("_Descrição: A matriz de confusão mostra a quantidade de acertos e erros do modelo para cada categoria de classificação._", unsafe_allow_html=True)

        # Aplicar o modelo a todo o conjunto de dados médicos
        df_ml['Classificação Prevista'] = pipeline.predict(df_ml['Observação Processada'])
        df_ml['Correto'] = df_ml['Classificação Ajustada'] == df_ml['Classificação Prevista']

        # Gerar a justificativa
        def generate_justificativa(row):
            if not row['Correto']:
                if row['Status agenda'] in ['Falta não justificada', 'Falta justificada']:
                    return 'Além de agendarem errado, o paciente faltou.'
                else:
                    return 'Classificação atual e prevista não coincidem.'
            else:
                return ''

        df_ml['Justificativa'] = df_ml.apply(generate_justificativa, axis=1)

        # Combinar os dados médicos com os de enfermagem
        df_final = pd.concat([df_ml, df[df['Tipo Profissional'] == 'Enfermeira']])
    else:
        st.warning("Não há dados suficientes para treinar o modelo de classificação.")
        df_final = df

    # Retornar 'df' para manter consistência nos dados usados nas análises
    return df

def analise_exploratoria(df):
    st.info("Realizando análise exploratória de dados...")

    # Análise por profissional
    st.subheader("Análise por Profissional")
    profissional_counts = df['Tipo Profissional'].value_counts()
    fig_profissional = px.bar(profissional_counts, x=profissional_counts.index, y=profissional_counts.values, labels={'x': 'Tipo Profissional', 'y': 'Quantidade'})
    st.plotly_chart(fig_profissional, use_container_width=True)
    st.markdown("_Descrição: Este gráfico mostra a quantidade de atendimentos realizados por tipo de profissional (Médico ou Enfermeira)._", unsafe_allow_html=True)

    # Análise por Classificação (Vagas Ocupadas)
    st.subheader("Análise por Classificação (Vagas Ocupadas)")
    df_ocupadas = df[df['Ocupado'] == 'Ocupado']
    classificacao_counts = df_ocupadas['Classificação Ajustada'].value_counts()
    fig_classificacao = px.bar(classificacao_counts, x=classificacao_counts.index, y=classificacao_counts.values, labels={'x': 'Classificação', 'y': 'Quantidade'})
    st.plotly_chart(fig_classificacao, use_container_width=True)
    st.markdown("_Descrição: Este gráfico mostra a distribuição das consultas ocupadas por classificação._", unsafe_allow_html=True)

    # Análise Temporal
    st.subheader("Análise Temporal")
    dia_counts = df['Dia da Semana'].value_counts()
    fig_dia = px.bar(dia_counts, x=dia_counts.index, y=dia_counts.values, labels={'x': 'Dia da Semana', 'y': 'Quantidade'})
    st.plotly_chart(fig_dia, use_container_width=True)
    st.markdown("_Descrição: Este gráfico mostra a quantidade de consultas realizadas em cada dia da semana._", unsafe_allow_html=True)

    # Taxa de Faltas por Classificação
    st.subheader("Taxa de Faltas por Classificação")
    df_ocupadas['Falta'] = df_ocupadas['Status agenda'].isin(['Falta não justificada', 'Falta justificada'])
    falta_rates = df_ocupadas.groupby('Classificação Ajustada')['Falta'].mean().reset_index()
    falta_rates['Falta'] = falta_rates['Falta'] * 100
    fig_falta = px.bar(falta_rates, x='Classificação Ajustada', y='Falta', labels={'Falta': 'Taxa de Faltas (%)'})
    st.plotly_chart(fig_falta, use_container_width=True)
    st.markdown("_Descrição: Este gráfico mostra a taxa de faltas (%) para cada classificação de consulta, ajudando a identificar em quais tipos de consultas os pacientes mais faltam._", unsafe_allow_html=True)

    # Análise de Sintomas mais Frequentes (considerando apenas vagas ocupadas)
    st.subheader("Sintomas Mais Frequentes")
    df_ocupadas = df[df['Ocupado'] == 'Ocupado']
    texto_todos = ' '.join(df_ocupadas['Observação Processada'])
    palavras = texto_todos.split()
    contador = Counter(palavras)

    # Remover palavras indesejadas
    palavras_ignoradas = ['dra', 'unimed', 'aut', 'ana', 'enf', 'sheil', 'letic', 'aspmj', 'mari', 'consult', 'vaness', 'ok', 'enfª', 'dr', 'syngo']
    for palavra in palavras_ignoradas:
        if palavra in contador:
            del contador[palavra]

    palavras_comuns = contador.most_common(20)
    palavras_df = pd.DataFrame(palavras_comuns, columns=['Palavra', 'Frequência'])
    fig_palavras = px.bar(palavras_df, x='Palavra', y='Frequência')
    st.plotly_chart(fig_palavras, use_container_width=True)
    st.markdown("_Descrição: Este gráfico mostra as 20 palavras mais frequentes nas observações das vagas ocupadas, indicando os sintomas mais comuns relatados pelos pacientes._", unsafe_allow_html=True)

    # Word Cloud
    st.subheader("Nuvem de Palavras dos Sintomas")
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=palavras_ignoradas).generate(texto_todos)
    fig_wc = plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(fig_wc)
    st.markdown("_Descrição: A nuvem de palavras visualiza os sintomas mais frequentes relatados nas observações das vagas ocupadas, com palavras maiores representando maior frequência._", unsafe_allow_html=True)

    # Gráfico de Linhas por Classificação e Hora do Dia, dividido por Dias da Semana
    st.subheader("Atendimentos por Classificação e Hora do Dia")
    dias_semana = ['Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 'Sexta-feira']
    dias_presentes = [dia for dia in dias_semana if dia in df['Dia da Semana'].unique()]
    tabs = st.tabs(dias_presentes)

    for i, dia in enumerate(dias_presentes):
        with tabs[i]:
            st.markdown(f"### {dia}")
            df_dia = df[(df['Dia da Semana'] == dia) & (df['Tipo Profissional'] == 'Médico')]
            if df_dia.empty:
                st.write(f"Sem dados para {dia}.")
                continue
            pivot_df = df_dia.pivot_table(index='Hora', columns='Classificação Ajustada', aggfunc='size', fill_value=0)
            pivot_df = pivot_df.reset_index()
            fig_linhas = px.line(pivot_df, x='Hora', y=pivot_df.columns[1:], labels={'Hora': 'Hora do Dia', 'value': 'Quantidade de Consultas', 'variable': 'Classificação'}, line_shape='spline', markers=True)
            st.plotly_chart(fig_linhas, use_container_width=True)
            st.markdown("_Descrição: Este gráfico mostra o número de atendimentos médicos por classificação e hora do dia._", unsafe_allow_html=True)

    # Análise de Ocupação das Vagas por Dia, Hora e Classificação (Médicos)
    st.subheader("Análise de Ocupação das Vagas por Classificação (Médicos)")
    classificacoes_medicas = ['Demanda espontânea', 'Programada', 'Urgência']
    tabs_classificacao = st.tabs(classificacoes_medicas)

    for i, classificacao in enumerate(classificacoes_medicas):
        with tabs_classificacao[i]:
            st.markdown(f"### Classificação: {classificacao}")
            df_classificacao = df[(df['Tipo Profissional'] == 'Médico') & (df['Classificação Ajustada'] == classificacao)]
            ocupacao = df_classificacao.groupby(['Dia da Semana', 'Hora', 'Ocupado']).size().unstack(fill_value=0).reset_index()
            total_vagas = ocupacao[['Livre', 'Ocupado', 'Bloqueado']].sum(axis=1)
            ocupacao['Total Vagas'] = total_vagas
            ocupacao['Taxa Ocupação'] = (ocupacao['Ocupado'] / ocupacao['Total Vagas']) * 100

            # Criar pivot table para o heatmap
            heatmap_data = ocupacao.pivot_table(index='Hora', columns='Dia da Semana', values='Taxa Ocupação')

            # Ordenar os dias da semana presentes nos dados
            dias_presentes = [dia for dia in dias_semana if dia in heatmap_data.columns]
            heatmap_data = heatmap_data[dias_presentes]

            if heatmap_data.empty:
                st.write(f"Sem dados para a classificação {classificacao}.")
                continue

            # Plotar heatmap
            fig_heatmap = plt.figure(figsize=(8,6))
            sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap='coolwarm', cbar_kws={'label': 'Taxa de Ocupação (%)'})
            plt.title(f'Taxa de Ocupação - {classificacao}')
            plt.xlabel('Dia da Semana')
            plt.ylabel('Hora do Dia')
            plt.tight_layout()
            st.pyplot(fig_heatmap)
            st.markdown("_Descrição: Este heatmap mostra a taxa de ocupação das vagas médicas para a classificação específica ao longo do dia e da semana._", unsafe_allow_html=True)

    # Períodos Críticos e Recomendações por Classificação
    st.subheader("Períodos Críticos e Recomendações por Classificação (Médicos)")

    for classificacao in classificacoes_medicas:
        st.markdown(f"### Classificação: {classificacao}")
        df_classificacao = df[(df['Tipo Profissional'] == 'Médico') & (df['Classificação Ajustada'] == classificacao)]
        ocupacao = df_classificacao.groupby(['Dia da Semana', 'Hora', 'Ocupado']).size().unstack(fill_value=0).reset_index()
        total_vagas = ocupacao[['Livre', 'Ocupado', 'Bloqueado']].sum(axis=1)
        ocupacao['Total Vagas'] = total_vagas
        ocupacao['Taxa Ocupação'] = (ocupacao['Ocupado'] / ocupacao['Total Vagas']) * 100

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Períodos com 100% de ocupação das vagas médicas:**")
            periodos_criticos = ocupacao[ocupacao['Taxa Ocupação'] == 100]
            if not periodos_criticos.empty:
                st.dataframe(periodos_criticos[['Dia da Semana', 'Hora', 'Total Vagas']])
                st.markdown("_Recomendação: Considerar a disponibilização de mais vagas nos horários acima para atender à demanda._", unsafe_allow_html=True)
            else:
                st.write("Não há períodos com 100% de ocupação total das vagas médicas para esta classificação.")

        with col2:
            st.write("**Períodos com baixa ocupação das vagas médicas (menos de 30%):**")
            periodos_baixa_ocupacao = ocupacao[ocupacao['Taxa Ocupação'] < 30]
            if not periodos_baixa_ocupacao.empty:
                st.dataframe(periodos_baixa_ocupacao[['Dia da Semana', 'Hora', 'Total Vagas', 'Taxa Ocupação']])
                st.markdown("_Recomendação: Avaliar a possibilidade de redistribuir as vagas não utilizadas para outras classificações ou horários de maior demanda._", unsafe_allow_html=True)
            else:
                st.write("Não há períodos com baixa ocupação das vagas médicas para esta classificação.")

def preprocessar_texto(texto):
    if pd.isnull(texto) or not isinstance(texto, str):
        return ''
    # Converter para minúsculas
    texto = texto.lower()

    # Remover números e caracteres especiais
    texto = re.sub(r'\d+', '', texto)
    texto = re.sub(r'[^\w\s]', '', texto)

    # Tokenização usando wordpunct_tokenize
    tokens = wordpunct_tokenize(texto)

    # Remover stopwords
    stop_words = set(stopwords.words('portuguese'))
    # Adicionar palavras a serem ignoradas
    palavras_ignoradas = ['dra', 'unimed', 'aut', 'ana', 'enf', 'sheil', 'letic', 'aspmj', 'mari', 'consult', 'vaness', 'ok', 'enfª', 'dr', 'syngo']
    stop_words.update(palavras_ignoradas)
    tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    stemmer = nltk.stem.RSLPStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    # Juntar tokens de volta em uma string
    texto_processado = ' '.join(tokens)

    return texto_processado

if __name__ == "__main__":
    main()