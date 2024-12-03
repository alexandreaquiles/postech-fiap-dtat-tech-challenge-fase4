import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# métricas
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# modelos
from prophet import Prophet
import xgboost as xgb

st.set_page_config(page_title='Análises do preço do petróleo Brent', page_icon=':oil_drum:', layout="wide")

@st.cache_data
def load_data():
  df = pd.read_html('http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view', skiprows=1)[0]
  df.columns = ['Data', 'Preco_Petroleo']
  df['Preco_Petroleo'] = df['Preco_Petroleo'] / 100
  df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y')
  df = df.set_index('Data')
  df = df.sort_values(by='Data', ascending=True)
  return df

df = load_data()

# Cabeçalho
st.write("# Pós Tech - Data Analytics - 5DTAT")
st.write("## Tech Challenge Fase 4 - Data viz and production models")
st.write('##### Análise Histórica e Forecasting do Preço do Petróleo Brent')
st.write('Código disponível em: https://github.com/alexandreaquiles/postech-fiap-dtat-tech-challenge-fase4')
st.divider()

aba1, aba2, aba3, aba4, aba5 = st.tabs(['Análise Histórica', 'Comparação de Modelos Preditivos', 'Plano de Deploy do Modelo em Produção', 'Dados brutos', 'Quem somos nós?'])

with aba4:
    #st.write(f'Dados históricos diários de {df.at(0,'Data')} a {df.at(-1,'Data')}')
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