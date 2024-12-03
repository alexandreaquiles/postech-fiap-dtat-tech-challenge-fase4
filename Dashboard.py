from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# métricas
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# modelos
from prophet import Prophet
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

df = carrega_dados()

date_format = '%d/%m/%Y'

datas_str = df.index.strftime(date_format)
primeira_data_str = datas_str.values[0]
ultima_data_str = datas_str.values[-1]

eventos_historicos = [
    {'date': '2008-09-15', 'y': 25, 'label': 'Quebra Lehman\nBrothers (15/09/2008)', 'xytext': (-140, 20)},
    {'date': '2014-11-27', 'y': 25, 'label': 'Petróleo xisto\nEUA (27/11/2014)', 'xytext': (-100, 20)},
    {'date': '2020-03-11', 'y': 25, 'label': 'COVID\n(11/03/2020)', 'xytext': (-100, 20)},
    {'date': '2022-02-24', 'y': 25,'label': 'Invasão\nUcrânia\n(24/02/2022)', 'xytext': (50, 20)}
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

aba1, aba2, aba3, aba4, aba5 = st.tabs(['Análise Histórica', 'Comparação de Modelos Preditivos', 'Plano de Deploy do Modelo em Produção', 'Dados brutos', 'Quem somos nós?'])

with aba1:
  
  fig_precos = px.line(df, y='Preco_Petroleo')

  for evento in eventos_historicos:
    fig_precos.add_shape(
        type="line",
        x0=evento['date'], x1=evento['date'],
        y0=df['Preco_Petroleo'].min(), 
        y1=df['Preco_Petroleo'].max(),
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

Aqui temos 4 exemplos disso:
          ''')

  st.write('''##### 1. Impacto da Quebra do Lehman Brothers (2008)

- **Crise financeira global**: A falência do Lehman Brothers levou a uma recessão mundial, diminuindo a demanda por petróleo.
- **Queda de preços**: A desaceleração econômica global fez com que o preço do petróleo caísse drasticamente, de mais de 140 USD/barril para menos de 40 USD/barril.
''')

  st.write('''##### 2. Petróleo de Xisto nos EUA (2016)

- **Aumento da produção**: A partir de 2010, a revolução do petróleo de xisto nos EUA aumentou significativamente a oferta de petróleo global, reduzindo os preços.
- **Impacto no mercado global**: A produção de petróleo de xisto tornou os EUA menos dependentes de importações e mais competitivos, causando uma queda nos preços do petróleo a partir de 2014, quando o petróleo foi de 100 USD/barril para abaixo de 30 USD/barril em 2016.
''')

  st.write('''##### 3. Impacto da COVID-19 (2020)

- **Queda na demanda**: Com a paralisação global de economias, transporte e indústria, a demanda por petróleo despencou (cerca de 30%).
- **Preços negativos**: Em abril de 2020, o preço do petróleo WTI caiu abaixo de zero devido ao excesso de oferta e falta de capacidade de armazenamento.
- **Recuperação gradual**: Com o avanço das vacinas e a reabertura das economias, a demanda foi se recuperando, mas de forma lenta e desigual.
''')

  st.write('''##### 4. Impacto da Guerra da Ucrânia (2022)

- **Redução da oferta**: Sanções contra a Rússia, um dos maiores produtores de petróleo, reduziram a oferta de petróleo no mercado.
- **Aumento de preços**: A incerteza sobre o fornecimento e o aumento da demanda por energia em meio ao conflito causaram alta nos preços.
- **Busca por alternativas**: Outros países, como os EUA, aumentaram a produção de petróleo de xisto para tentar compensar a falta da oferta russa.
''')
with aba4:
    st.write(f'Dados históricos diários de { primeira_data_str } a { ultima_data_str }')
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