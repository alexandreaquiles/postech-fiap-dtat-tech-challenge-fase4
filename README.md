# postech-fiap-dtat-tech-challenge-fase4

Entrega do Tech Challenge da Fase 4 da Pós Tech em Data Analytics da FIAP, turma 5DTAT.

## Análise Histórica e Forecasting do Preço do Petróleo Brent

Neste relatório, realizamos uma análise histórica e um forecasting do preço do barril de pretróleo Brent.

A fonte dos dados é o site do Ipea: http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view

Desenvolvemos um Dashboard Interativo que torna possível gerar insights para a tomada de decisão. Além disso, desenvolvemos um modelo de Machine Learning preditivo, com o intuito de fazer o _forecasting_ do preço do petróleo.

O Dashboard pode ser encontrado em: https://dtat5-tech-challenge-fase4-grupo46-petroleo-brent.streamlit.app/

### Quem somos nós?

Somos da Consultoria **Grupo 46**, uma consultoria especializada em análise histórica e forecasting de commodities e produtos financeiros.

Integrantes:

- Alexandre Aquiles Sipriano da Silva (alexandre.aquiles@gmail.com)
- Gabriel Machado Costa (gabrielmachado2211@gmail.com)
- Caio Martins Borges (caio.borges@bb.com.br)

## Detalhes técnicos

A análise exploratória dos dados, testes estatíscos, comparativos de modelos e os resultados podem ser encontrados no seguinte notebook: [5dtat_tech_challenge_fase4_petroleo_brent.ipynb](https://github.com/alexandreaquiles/postech-fiap-dtat-tech-challenge-fase4/blob/main/5dtat_tech_challenge_fase4_petroleo_brent.ipynb)

Utilizamos **Matplotlib** para visualizações de dados com o intuito de auxiliar na análise exploratória.

Utilizando o **Pandas**, exploramos estatísticas rolantes (média móvel e desvio padrão) com janela de 12 dias.

Utilizamos a biblioteca **statsmodel** para decompar a série temporal em tendência, sazonalidade e resíduos. Além disso, fizemos o teste de Dickey-Fuller para verificar estacionariedade, concluindo pela hipótese nula.

Fizemos modelos preditivos, considerando 30 dias para teste e o restante para treino utlizando as seguintes bibliotecas:

- ARIMA e SARIMAX com statsmodel 
- AutoARIMA com statsforecast
- Prophet
- XGBoost

Utilizamos as métricas do **sklearn** para comparação dos modelos preditivos. Para o Dashboard, utilizamos o **Streamlit**, gerando gráficos interativos com **Plotly Express**.

### Conclusões

Os resultados foram os seguintes:

| Modelo | MAE | MSE | MAPE |
|---|---|---|---|
| XGBoost | 2.845920 | 10.904850 | 3.867445 |
| AutoARIMA | 4.953615 | 26.320127 | 6.722812 |
| ARIMA | 5.846990 | 35.997389 | 7.929802 |
| SARIMAX | 6.047026 | 38.331656	 | 8.199489 |
| Prophet | 15.691890 | 248.811764 | 21.228882 |

Os melhores resultados foram do modelo **XGBoost**, que foi o considerado para o Dashboard Interativo e o deploy em produção.

## Próximos passos

Modelos como o XGBoost, Prophet e SARIMAX aumentam sua performance quando são utilizadas variáveis exógenas. Como passos a explorar poderíamos considerar dados externos como como:

- **DXY (US Dollar Index)**: O DXY (Índice do Dólar dos EUA) é uma medida do valor do dólar americano em relação a uma cesta de seis principais moedas mundiais: Euro, Iene Japonês, Libra Esterlina, Dólar Canadense, Coroa Sueca e Franco Suíço. É um indicador-chave da força do dólar nos mercados globais e frequentemente utilizado por traders e economistas para avaliar o desempenho das moedas.
- **S&P 500**: O Standard & Poor's 500, é um índice de mercado que acompanha 500 grandes empresas de capital aberto listadas nas bolsas de valores dos Estados Unidos. É considerado um indicador-chave do desempenho geral do mercado acionário americano e da economia dos EUA como um todo. Investidores e economistas frequentemente o utilizam como referência para avaliar a saúde do mercado e as tendências econômicas.
- **WTI (West Texas Intermediate)**: WTI (West Texas Intermediate) é um tipo específico de petróleo bruto utilizado como referência na precificação do petróleo. É produzido nos Estados Unidos, principalmente no Texas e no Novo México, e é conhecido por sua alta qualidade e baixo teor de enxofre. O WTI é negociado na Bolsa de Mercadorias de Nova York (NYMEX) e serve como um preço de referência importante para os mercados de petróleo, frequentemente comparado ao Brent, extraído no Mar do Norte.

Esses dados podem ser obtidos com a biblioteca `yfinance` com os seguintes tickers:

- `DX-Y.NYB` para o indíce comparativo do dólar americano DXY
- `^GSPC` para o índice da bolsa de valores dos EUA Standard & Poor's 500
- `CL=F` para o petróleo bruto WTI

### Como rodar localmente?

Para rodar localmente, é necessário ter o Python 3+ e instalar todas as dependências com o comando:

```sh
pip install -r requirements.txt
```

Em seguida, precisamos executar a CLI do Streamlit:

```sh
streamlit run Dashboard.py
```
