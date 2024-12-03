from datetime import datetime
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# métricas
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# modelos
import xgboost as xgb

st.set_page_config(page_title='Análises do preço do petróleo Brent', page_icon=':oil_drum:', layout="wide")

@st.cache_data
def carrega_dados():
  df = pd.read_html('http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view', skiprows=1)[0]
  df.columns = ['Data', 'Preco_Petroleo']
  df['Preco_Petroleo'] = df['Preco_Petroleo'] / 100
  df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y')
  df = df.set_index('Data')
  df = df.sort_values(by='Data', ascending=True)
  return df

@st.cache_resource
def predict_xgboost(train, test):

  # Preparando dados de treino para o XGBoost
  train_xgb = train.copy()
  train_xgb = train_xgb.reset_index()
  train_xgb['ano'] = train_xgb['Data'].dt.year
  train_xgb['mes'] = train_xgb['Data'].dt.month
  train_xgb['dia'] = train_xgb['Data'].dt.day
  train_xgb['diadasemana'] = train_xgb['Data'].dt.dayofweek

  # Preparando dados de teste para o XGBoost
  test_xgb = test.copy()
  test_xgb = test_xgb.reset_index()
  test_xgb['ano'] = test_xgb['Data'].dt.year
  test_xgb['mes'] = test_xgb['Data'].dt.month
  test_xgb['dia'] = test_xgb['Data'].dt.day
  test_xgb['diadasemana'] = test_xgb['Data'].dt.dayofweek
  
  FEATURES = ['ano', 'mes', 'dia', 'diadasemana']
  TARGET = 'Preco_Petroleo'

  X_train_xgb, y_train_xgb = train_xgb[FEATURES], train_xgb[TARGET]
  X_test_xgb, y_test_xgb = test_xgb[FEATURES], test_xgb[TARGET]

  reg = xgb.XGBRegressor(objective='reg:squarederror')
  reg.fit(X_train_xgb, y_train_xgb)

  predict_xgb = reg.predict(X_test_xgb)

  return test_xgb, predict_xgb

def plot_testpredict(model, x_test, y_test, x_predict, y_predict):
  fig = px.line()

  fig.add_scatter(x=x_test, y=y_test, name='Dados de Teste')
  fig.add_scatter(x=x_predict, y=y_predict, name=f'Previsões {model}')

  fig.update_layout(
      title=f'Previsão do Valor de Fechamento do Índice da Bolsa - {model}',
      xaxis_title='Data',
      yaxis_title='Valor de Fechamento',
      height=500
  )

  return fig

def calculate_metrics(y_true, y_pred):
  mae = mean_absolute_error(y_true, y_pred)
  mse = mean_squared_error(y_true, y_pred)
  mape = mean_absolute_percentage_error(y_true, y_pred) * 100
  return mae, mse, mape

df = carrega_dados()

date_format = '%d/%m/%Y'

min_date = df.index[0].to_pydatetime()
max_date = df.index[-1].to_pydatetime()

default_test_size = 30

eventos_historicos = [
    {'date': '2008-09-15', 'y': 25, 'label': 'Quebra Lehman\nBrothers (15/09/2008)', 'xytext': (-140, 20), 'descricao': '''##### Impacto da Quebra do Lehman Brothers (2008)

- **Crise financeira global**: A falência do Lehman Brothers levou a uma recessão mundial, diminuindo a demanda por petróleo.
- **Queda de preços**: A desaceleração econômica global fez com que o preço do petróleo caísse drasticamente, de mais de 140 USD/barril para menos de 40 USD/barril.
'''},
    {'date': '2014-11-27', 'y': 25, 'label': 'Petróleo xisto\nEUA (27/11/2014)', 'xytext': (-100, 20), 'descricao': '''##### Petróleo de Xisto nos EUA (2016)

- **Aumento da produção**: A partir de 2010, a revolução do petróleo de xisto nos EUA aumentou significativamente a oferta de petróleo global, reduzindo os preços.
- **Impacto no mercado global**: A produção de petróleo de xisto tornou os EUA menos dependentes de importações e mais competitivos, causando uma queda nos preços do petróleo a partir de 2014, quando o petróleo foi de 100 USD/barril para abaixo de 30 USD/barril em 2016.
'''},
    {'date': '2020-03-11', 'y': 25, 'label': 'COVID\n(11/03/2020)', 'xytext': (-100, 20), 'descricao': '''##### Impacto da COVID-19 (2020)

- **Queda na demanda**: Com a paralisação global de economias, transporte e indústria, a demanda por petróleo despencou (cerca de 30%).
- **Preços negativos**: Em abril de 2020, o preço do petróleo WTI caiu abaixo de zero devido ao excesso de oferta e falta de capacidade de armazenamento.
- **Recuperação gradual**: Com o avanço das vacinas e a reabertura das economias, a demanda foi se recuperando, mas de forma lenta e desigual.
'''},
    {'date': '2022-02-24', 'y': 25,'label': 'Invasão\nUcrânia\n(24/02/2022)', 'xytext': (50, 20), 'descricao': '''##### Impacto da Guerra da Ucrânia (2022)

- **Redução da oferta**: Sanções contra a Rússia, um dos maiores produtores de petróleo, reduziram a oferta de petróleo no mercado.
- **Aumento de preços**: A incerteza sobre o fornecimento e o aumento da demanda por energia em meio ao conflito causaram alta nos preços.
- **Busca por alternativas**: Outros países, como os EUA, aumentaram a produção de petróleo de xisto para tentar compensar a falta da oferta russa.
'''}
]

# Cabeçalho
st.write("# Pós Tech - Data Analytics - 5DTAT")
st.write("## Tech Challenge Fase 4 - Data viz and production models")
st.write('### Análise Histórica e Forecasting do Preço do Petróleo Brent')
st.write('''Neste relatório, realizamos uma **Análise Histórica** e um **Modelo de Forecasting** do preço do barril de pretróleo Brent.

O petróleo Brent é um tipo de petróleo bruto produzido no Mar do Norte, na Europa, e serve como fator de comparação para o preço internacional de diferentes tipos de petróleo. Consideramos o preço do barril de petróleo tipo Brent em dólares (US$), não incluindo despesas com frete e seguro (preço FOB - free on board).
''')
st.write('Código disponível em: https://github.com/alexandreaquiles/postech-fiap-dtat-tech-challenge-fase4')
st.divider()

st.sidebar.title('Filtros')

start_date = st.sidebar.slider('Data Inicial', min_date, max_date, max_date - relativedelta(years=5))
st.sidebar.caption(f'Data a partir da qual queremos as previsões. Dados disponíveis de {min_date.strftime("%d/%m/%Y")} a {max_date.strftime("%d/%m/%Y")}.')

if start_date:
    df_filtrado = df[df.index >= start_date]

test_size = st.sidebar.number_input('Tamanho do teste', min_value=1, max_value=180, value=default_test_size)
st.sidebar.caption(f'Tamanho do dataset separado para teste dos modelos. Por padrão, será considerado {default_test_size}.')

aba1, aba2, aba3, aba4, aba5 = st.tabs(['Análise Histórica', 'Modelo Preditivo com XGBoost', 'Plano de Deploy do Modelo em Produção', 'Dados brutos', 'Quem somos nós?'])

# criando dfs de treino e teste
df_modeling = df_filtrado[['Preco_Petroleo']]
train_size = df_modeling.shape[0] - test_size
train = df_modeling[:train_size]
test = df_modeling[train_size:]

with aba1:

  eventos_historicos_filtrados = []
  for evento in eventos_historicos:

    evento_date = datetime.strptime(evento['date'], '%Y-%m-%d')
    min_date_filtrado = df_filtrado.index[0].to_pydatetime()
 
    if evento_date >= min_date_filtrado:
      eventos_historicos_filtrados.append(evento)
      
  fig_precos = px.line(df_filtrado, y='Preco_Petroleo')

  for evento in eventos_historicos_filtrados:
    fig_precos.add_shape(
        type="line",
        x0=evento['date'], x1=evento['date'],
        y0=df_filtrado['Preco_Petroleo'].min(), 
        y1=df_filtrado['Preco_Petroleo'].max(),
        line=dict(color="gray", width=1, dash="dash")
    )
    
    fig_precos.add_annotation(
        x=evento['date'],
        y=evento['y'],
        text=evento['label'],
        showarrow=True,
        arrowhead=1,
        arrowcolor='gray',
        ax=evento['xytext'][0], 
        ay=evento['xytext'][1]
    )

  fig_precos.update_layout(
    title='Preços históricos do Pretróleo Brent',
    xaxis_title='Data',
    yaxis_title='Preço Brent (US$)',
    xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray', type='date'),
    yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
  )
   
  st.plotly_chart(fig_precos, use_container_width=True)
  
  st.write('#### Resumo dos Impactos Globais no Preço do Petróleo Brent')

  st.write('''O preço do petróleo é **altamente sensível** a **movimentos globais**, como **oferta e demanda** (impactados por crises, mudanças econômicas e novas fontes de energia), **conflitos geopolíticos** (como guerras e sanções) e **expectativas do mercado** (influenciadas por incertezas e especulação).

Esses eventos refletem como fatores econômicos, políticos e tecnológicos globais podem afetar os preços do petróleo, que são fundamentais para a economia mundial.
          ''')

  if eventos_historicos_filtrados:
    st.write(f'Aqui temos {len(eventos_historicos_filtrados)} exemplos disso:')
    for evento in eventos_historicos_filtrados:
      st.write(evento['descricao'])

with aba2:
  test_xgb, predict_xgb = predict_xgboost(train, test)
  metrics_xgb = calculate_metrics(test_xgb['Preco_Petroleo'], predict_xgb)

  st.subheader('Métricas de Avaliação')
  df_metrics = pd.DataFrame(
    [metrics_xgb],
    columns=['MAE', 'MSE', 'MAPE'],
    index=['XGBoost'],
  )
  st.dataframe(df_metrics, use_container_width=True, column_config={
      'MSE': st.column_config.NumberColumn(format="%d"),
  })

  st.subheader('Gráfico')
  fig_xbg = plot_testpredict('XGBoost', test_xgb['Data'], test_xgb['Preco_Petroleo'], test_xgb['Data'], predict_xgb)
  st.plotly_chart(fig_xbg, use_container_width=True)

with aba4:
  st.write(f'Dados históricos diários de { min_date.strftime(date_format) } a { max_date.strftime(date_format) }')
  st.write('Fonte: http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view')
  st.dataframe(df, use_container_width=True, column_config={
    '_index': st.column_config.DatetimeColumn(format='DD/MM/YYYY'),
    'Preco_Petroleo': st.column_config.NumberColumn(label='Preço Brent (US$)', format="%f")
  })

with aba5:
  st.write('''###### Grupo 46

  Somos da Consultoria Grupo 46, uma consultoria especializada em análise histórica e forecasting de commodities e produtos financeiros.

  Integrantes: 

  * Alexandre Aquiles Sipriano da Silva (alexandre.aquiles@gmail.com)
  * Gabriel Machado Costa (gabrielmachado2211@gmail.com)
  * Caio Martins Borges (caio.borges@bb.com.br)
  ''')