# Datathon — Plataforma Adaptativa de Ofertas com Contextual Thompson Sampling

## Sumário

* [1. Visão geral](#1-visão-geral)
* [2. Objetivo de negócio](#2-objetivo-de-negócio)
* [3. Base de dados](#3-base-de-dados)
* [4. Análise exploratória](#4-análise-exploratória)
* [5. Feature Engineering](#5-feature-engineering)
* [6. Variáveis utilizadas pelo modelo](#6-variáveis-utilizadas-pelo-modelo)
* [7. Preparação dos dados](#7-preparação-dos-dados)
* [8. Baseline](#8-baseline)
* [9. Contextual Thompson Sampling](#9-contextual-thompson-sampling)
* [10. Aprendizado offline](#10-aprendizado-offline)
* [11. Comparação com o baseline](#11-comparação-com-o-baseline)
* [12. Estrutura do projeto](#12-estrutura-do-projeto)
* [13. Pipeline do projeto](#13-pipeline-do-projeto)
* [14. Como executar o projeto](#14-como-executar-o-projeto)
  * [14.1. Clonar o repositório](#141-clonar-o-repositório)
  * [14.2. Criar ambiente virtual](#142-criar-ambiente-virtual)
  * [14.3. Instalar dependências](#143-instalar-dependências)
* [15. Treinar o modelo oficial](#15-treinar-o-modelo-oficial)
* [16. MLflow](#16-mlflow)
* [17. Testar a inferência](#17-testar-a-inferência)
* [18. Executar a API](#18-executar-a-api)
* [19. Considerações sobre as features](#19-considerações-sobre-as-features)
* [20. Limitações](#20-limitações)
* [21. Produtização da arquitetura em AWS](#21-produtização-da-arquitetura-em-aws)
  * [21.1. Objetivo da arquitetura](#211-objetivo-da-arquitetura)
  * [21.2. Arquitetura proposta](#212-arquitetura-proposta)
  * [21.3. Fluxo de dados](#213-fluxo-de-dados)
  * [21.4. Componentes AWS](#214-componentes-aws)
  * [21.5. Treinamento e atualização do modelo](#215-treinamento-e-atualização-do-modelo)
  * [21.6. Inferência online](#216-inferência-online)
  * [21.7. Atualização online do Bandit](#217-atualização-online-do-bandit)
  * [21.8. Monitoramento e observabilidade](#218-monitoramento-e-observabilidade)
  * [21.9. Segurança e governança](#219-segurança-e-governança)
  * [21.10. Escalabilidade](#2110-escalabilidade)
  * [21.11. Arquitetura resumida](#2111-arquitetura-resumida)
* [22. Conclusão](#22-conclusão)

---

# 1. Visão geral

Este projeto foi desenvolvido como solução para o **Datathon**, com o objetivo de construir uma plataforma de experimentação adaptativa capaz de selecionar o melhor canal de abordagem para cada cliente com base em seu contexto e no histórico de conversões.

A documentação original do **Tech Challenge** também está disponível no diretório /docs.

O problema é formulado como um **Contextual Multi-Armed Bandit**, no qual o modelo precisa decidir qual ação realizar para cada cliente, equilibrando:

* **Exploração:** testar ações para obter novas informações;
* **Explotação:** utilizar o conhecimento adquirido para escolher ações com maior recompensa esperada;
* **Personalização:** considerar as características do cliente na decisão;
* **Aprendizado contínuo:** atualizar o modelo conforme novas recompensas são observadas.

Neste projeto, foram utilizadas duas ações:

* `cellular` — contato por celular;
* `telephone` — contato por telefone.

A variável `y` representa o resultado observado da campanha:

* `1` — conversão;
* `0` — não conversão.

O algoritmo escolhido foi o **Contextual Thompson Sampling**, que utiliza o contexto do cliente para estimar a recompensa esperada de cada ação e realizar a seleção de forma probabilística.

---

# 2. Objetivo de negócio

O objetivo é demonstrar como uma instituição financeira poderia substituir uma política fixa de escolha de canal por uma estratégia adaptativa.

Em vez de utilizar uma regra como:

> "Sempre utilizar o canal com maior conversão histórica."

o modelo recebe as características do cliente e avalia as ações disponíveis:

```text
Cliente
   │
   ├── Características demográficas
   ├── Características comportamentais
   ├── Histórico de contatos
   ├── Características temporais
   └── Indicadores econômicos
            │
            ▼
   Contextual Thompson Sampling
            │
       ┌────┴────┐
       ▼         ▼
   cellular   telephone
       │         │
       └────┬────┘
            ▼
       ação escolhida
```

O objetivo não é apenas prever a probabilidade de conversão, mas **escolher uma ação levando em consideração a incerteza e o trade-off entre exploração e explotação**.

---

# 3. Base de dados

Foi utilizada a base **Bank Marketing**, disponibilizada no Kaggle, contendo informações de campanhas de marketing de uma instituição bancária.

A base contém características dos clientes, informações sobre contatos realizados, variáveis temporais e indicadores econômicos, além do resultado da campanha.

A referência da base e sua documentação estão disponíveis no diretório /docs.

---

# 4. Análise exploratória

A Análise Exploratória de Dados (EDA) foi realizada no notebook:

```text
notebooks/01_EDA.ipynb
```

A análise teve como objetivo compreender:

* distribuição das variáveis;
* distribuição da variável alvo;
* características dos clientes;
* comportamento das conversões;
* relação entre características dos clientes e conversão;
* distribuição das ações históricas;
* possíveis problemas de qualidade dos dados;
* possíveis fontes de vazamento de informação;

---

# 5. Feature Engineering

O tratamento e a criação das variáveis foram realizados em:

```text
notebooks/02_FeatureEngineering.ipynb
```

Além das variáveis originais, foram criadas features derivadas para representar aspectos relevantes do cliente e de seu histórico.

Entre as principais features estão:

| Feature            | Descrição                                                     |
| ------------------ | ------------------------------------------------------------- |
| `age_group`        | Agrupamento da idade do cliente                               |
| `financial_risk`   | Indicador derivado relacionado às características financeiras |
| `engagement_score` | Indicador de engajamento/histórico do cliente                 |
| `previous_contact` | Indica existência de contato anterior                         |
| `previous_success` | Indica sucesso em contato anterior                            |

Essas variáveis foram utilizadas para enriquecer o contexto disponibilizado ao algoritmo.

---

# 6. Variáveis utilizadas pelo modelo

O modelo utiliza **23 features originais de contexto**:

```text
age
job
marital
education
default
housing
loan
month
day_of_week
campaign
pdays
previous
poutcome
emp.var.rate
cons.price.idx
cons.conf.idx
euribor3m
nr.employed
previous_contact
previous_success
age_group
financial_risk
engagement_score
```

A variável:

```text
contact
```

não é utilizada como feature.

Ela representa a **ação histórica**, ou seja, uma das arms disponíveis:

```text
cellular
telephone
```

Da mesma forma:

```text
y
```

não é utilizada como feature.

Ela representa o **reward observado**:

```text
y = 1 → conversão
y = 0 → não conversão
```

Essa separação é importante para evitar que o algoritmo utilize a própria ação ou o resultado observado como informação de entrada.

---

# 7. Preparação dos dados

O projeto utiliza diferentes tipos de variáveis.

As variáveis categóricas são transformadas utilizando **One-Hot Encoding**.

As variáveis numéricas são mantidas em sua forma numérica.

O encoder é ajustado somente com os dados de treinamento:

```text
X_train
   │
   ▼
OneHotEncoder
   │
   ▼
Contexto numérico
```

O mesmo encoder é posteriormente utilizado na inferência.

Isso garante que o contexto enviado para o modelo durante a API possua exatamente a mesma representação utilizada durante o treinamento.

Após o One-Hot Encoding e a adição do intercepto, o modelo trabalha atualmente com:

```text
70 dimensões de contexto
```

O `encoder` utilizado pelo modelo é salvo em:

```text
models/encoder.pkl
```

---

# 8. Baseline

O baseline foi desenvolvido no notebook:

```text
notebooks/03_Baseline.ipynb
```

O objetivo do baseline é estabelecer uma referência simples para verificar se o algoritmo adaptativo apresenta ganho.

A política do baseline utiliza a classe majoritária de conversão para cada perfil de cliente.

O resultado obtido pelo baseline foi:

| Métrica   |  Resultado |
| --------- | ---------: |
| Accuracy  | **0,8795** |
| Precision | **0,3502** |
| Recall    | **0,0819** |
| F1-score  | **0,1328** |
| ROC AUC   | **0,5313** |

O resultado evidencia a dificuldade de identificar a classe positiva em um problema desbalanceado.

---

# 9. Contextual Thompson Sampling

O algoritmo adaptativo foi desenvolvido inicialmente no notebook:

```text
notebooks/04_ThompsonSampling.ipynb
```

Posteriormente, a implementação foi transformada em código reutilizável em:

```text
src/bandit.py
```

O modelo possui duas arms:

```python
arms = [
    "cellular",
    "telephone"
]
```

Para cada cliente, o algoritmo recebe um vetor de contexto:

```text
x = características do cliente
```

e estima a recompensa esperada para cada ação.

A escolha da ação é probabilística, permitindo que o modelo explore alternativas enquanto utiliza o conhecimento adquirido.

---

# 10. Aprendizado offline

Antes da implementação da aplicação, foi realizado um experimento de aprendizado offline utilizando o histórico da base.

O conjunto de treinamento foi utilizado para inicializar os parâmetros do modelo e o conjunto de teste foi utilizado para avaliar a política aprendida.

No experimento, foram obtidos:

```text
Observações avaliadas: 8238
Ações coincidentes:    3831
Taxa de coincidência:  46,50%
```

A política aprendida apresentou a seguinte distribuição de ações:

| Ação        | Percentual |
| ----------- | ---------: |
| `cellular`  | **71,49%** |
| `telephone` | **28,51%** |

A distribuição não significa que o modelo sempre escolherá `cellular`. Como o Thompson Sampling realiza amostragem, a decisão pode variar conforme o contexto e a incerteza associada às estimativas.

---

# 11. Comparação com o baseline

No experimento offline realizado, foram observados os seguintes resultados:

| Modelo            | Observações avaliadas | Taxa de coincidência | Reward médio | Conversões observadas |
| ----------------- | --------------------: | -------------------: | -----------: | --------------------: |
| Baseline          |                  6240 |               75,75% |   **0,1212** |                   756 |
| Thompson Sampling |                  3831 |               46,50% |   **0,1441** |                   522 |

O resultado apresentou:

```text
Reward Baseline : 0,1212
Reward Thompson : 0,1441

Ganho absoluto  : 0,0229
Ganho relativo  : 18,93%
```

O Thompson Sampling apresentou um **reward médio aproximadamente 18,93% superior ao baseline** no experimento realizado.

A taxa de coincidência não deve ser interpretada como acurácia do modelo. Ela representa a proporção de casos em que a ação escolhida pelo Thompson Sampling coincidiu com a ação histórica registrada na base.

---

# 12. Estrutura do projeto

```text
tech_challenge_5/
│
├── app/
│   └── api.py
│
├── config/
│   └── settings.toml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│   ├── DatabaseBankMarketing.md
│   ├── TechChallenge.md
│   └── ThompsonSampling.md
│
├── examples/
│   └── get_post.py
|
├── mlflow/
│   └── mlflow.db
|
├── mlruns/
|
├── models/
│   ├── bandit.pkl
│   └── encoder.pkl
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_FeatureEngineering.ipynb
│   ├── 03_Baseline.ipynb
│   ├── 04_ThompsonSampling.ipynb
│   └── 05_Evaluation.md
│
├── scripts/
│   └── train.py
│
├── src/
│   ├── __init__.py
│   ├── bandit.py
│   ├── config.py
│   ├── inference.py
│   └── preprocessing.py
│
├── requirements.txt
└── README.md
```

---

# 13. Pipeline do projeto

O fluxo completo pode ser representado da seguinte forma:

```text
Base Kaggle
    │
    ▼
01_EDA.ipynb
    │
    ▼
02_FeatureEngineering.ipynb
    │
    ▼
data/processed/
    │
    ├──────────────────────┐
    ▼                      ▼
03_Baseline.ipynb     04_ThompsonSampling.ipynb
                           │
                           ▼
                       src/bandit.py
                           │
                           ▼
                       scripts/train.py
                           │
                  ┌────────┴────────┐
                  ▼                 ▼
            bandit.pkl          encoder.pkl
                  │                 │
                  └────────┬────────┘
                           ▼
                     src/inference.py
                           │
                           ▼
                        app/api.py
                           │
                           ▼
                       Recomendação
```

---

# 14. Como executar o projeto

## 14.1. Clonar o repositório

```bash
git clone <https://github.com/alicebiasilva/adaptive-marketing-offer-engine.git>
cd tech_challenge_5
```

---

## 14.2. Criar ambiente virtual

No Windows:

```bash
python -m venv .venv
```

Ative o ambiente:

```bash
.venv\Scripts\activate
```

---

## 14.3. Instalar dependências

Utilizando `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

# 15. Treinar o modelo oficial

Depois que os dados processados estiverem disponíveis, execute:

```bash
python scripts/train.py
```

O treinamento irá:

1. carregar os dados processados;
2. separar contexto, ação e reward;
3. dividir os dados em treino e teste;
4. identificar features categóricas e numéricas;
5. aplicar One-Hot Encoding;
6. gerar o contexto de 70 dimensões;
7. criar o Contextual Thompson Sampling;
8. realizar o treinamento offline;
9. salvar o modelo;
10. salvar o encoder;
11. registrar parâmetros e métricas no MLflow.

Os artefatos serão salvos em:

```text
models/
├── bandit.pkl
└── encoder.pkl
```

---

# 16. MLflow

O projeto utiliza MLflow para registrar os experimentos de treinamento.

Os experimentos são armazenados localmente em:

```text
mlruns/
```

Para iniciar a interface:

```bash
mlflow ui
```

Depois acesse:

```text
http://127.0.0.1:5000
```

O experimento utilizado é:

```text
contextual_thompson_sampling
```

Entre os parâmetros registrados estão:

* algoritmo;
* random state;
* tamanho do conjunto de teste;
* quantidade de features originais;
* quantidade de dimensões do contexto;
* quantidade de arms;
* arms utilizadas;
* valor de `alpha`;
* quantidade de features categóricas;
* quantidade de features numéricas.

Também são registradas métricas relacionadas ao treinamento e aos parâmetros estimados.

---

# 17. Testar a inferência

Depois do treinamento, execute:

```bash
python -m src.inference
```

O resultado esperado é semelhante a:

```text
============================================================
INFERÊNCIA
============================================================

Ação escolhida : cellular
Score          : 0.895500
```

O resultado pode variar entre execuções porque o Thompson Sampling realiza amostragem para representar a incerteza das estimativas.

---

# 18. Executar a API

Com os artefatos gerados, execute:

```bash
uvicorn app.api:app --reload
```

A API ficará disponível em:

```text
http://127.0.0.1:8000
```

A documentação interativa do FastAPI pode ser acessada em:

```text
http://127.0.0.1:8000/docs
```

A API recebe as características de um cliente e retorna a ação recomendada pelo modelo.

Exemplo conceitual:

```json
{
    "action": "cellular",
    "score": 0.895500
}
```

---

# 19. Considerações sobre as features

A escolha das features foi baseada na necessidade de representar diferentes dimensões do contexto do cliente.

### Perfil

```text
age
job
marital
education
age_group
```

Representam características demográficas e de perfil.

### Histórico

```text
previous
pdays
poutcome
previous_contact
previous_success
```

Representam o relacionamento e os resultados de contatos anteriores.

### Comportamento

```text
campaign
engagement_score
```

Representam aspectos relacionados ao comportamento e interação com campanhas.

### Temporal

```text
month
day_of_week
```

Capturam possíveis variações temporais nas campanhas.

### Econômico

```text
emp.var.rate
cons.price.idx
cons.conf.idx
euribor3m
nr.employed
```

Representam o contexto macroeconômico associado à observação.

---

# 20. Limitações

Este projeto utiliza uma base histórica de campanhas para demonstrar uma arquitetura de experimentação adaptativa.

Portanto, os resultados offline não representam necessariamente o comportamento de um sistema online em produção.

Em particular:

* a base contém decisões históricas;
* o modelo não possui feedback online real;
* a avaliação offline depende das ações observadas no histórico;
* a taxa de coincidência não representa acurácia;
* o reward observado está condicionado às ações que foram efetivamente realizadas;
* o experimento não deve ser interpretado como uma validação de impacto causal.

Além disso, variáveis com potencial de vazamento temporal devem ser analisadas cuidadosamente antes de utilização em um cenário real.

O objetivo principal é demonstrar a construção de um pipeline de **Machine Learning Engineering + Contextual Bandit**, incluindo preparação de dados, baseline, treinamento, inferência, API e acompanhamento de experimentos.

---

# 21. Produtização da arquitetura em AWS

A solução atual foi desenvolvida para execução local, com treinamento offline, persistência dos artefatos e uma API FastAPI para inferência.

Para uma utilização em ambiente produtivo, a arquitetura poderia ser evoluída para uma solução em nuvem utilizando serviços gerenciados da **AWS**, permitindo escalabilidade, monitoramento, segurança e atualização contínua do modelo.

A arquitetura proposta separa os principais componentes em:

* ingestão e armazenamento de dados;
* processamento e preparação;
* treinamento;
* registro e versionamento de modelos;
* inferência online;
* coleta de rewards;
* atualização do Contextual Bandit;
* monitoramento e observabilidade.

---

## 21.1. Objetivo da arquitetura

Em produção, o sistema deve ser capaz de:

1. receber o contexto de um cliente;
2. selecionar uma ação entre `cellular` e `telephone`;
3. retornar a recomendação para o sistema consumidor;
4. registrar a decisão tomada;
5. observar posteriormente o resultado da ação;
6. transformar o resultado em reward;
7. atualizar o modelo;
8. monitorar a qualidade da política;
9. detectar mudanças no comportamento dos dados;
10. permitir rollback para uma versão anterior do modelo.

O principal diferencial em relação ao ambiente atual seria a passagem de um processo predominantemente **offline** para um processo com **feedback contínuo**.

---

## 21.2. Arquitetura proposta

Uma arquitetura possível seria:

```text
                         ┌──────────────────────┐
                         │  Sistemas do Banco   │
                         │ CRM / Campanhas / App│
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      API Gateway     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ AWS Lambda / ECS      │
                         │ API de Recomendação  │
                         └──────────┬───────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     ▼                             ▼
             ┌───────────────┐             ┌───────────────┐
             │ Modelo Bandit │             │ Feature Store │
             │ Versionado    │             │ / Dados       │
             └───────┬───────┘             └───────────────┘
                     │
                     ▼
              Ação recomendada
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
      cellular              telephone
          │                     │
          └──────────┬──────────┘
                     ▼
             Resultado da ação
                     │
                     ▼
              ┌──────────────┐
              │ Amazon       │
              │ Kinesis      │
              └──────┬───────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
   ┌──────────────┐      ┌──────────────┐
   │ Amazon S3    │      │ Processamento │
   │ Data Lake    │      │ / ETL         │
   └──────┬───────┘      └──────┬───────┘
          │                     │
          └──────────┬──────────┘
                     ▼
             Treinamento / Update
                     │
                     ▼
             ┌──────────────┐
             │ MLflow /     │
             │ Model        │
             │ Registry     │
             └──────┬───────┘
                    │
                    ▼
             Nova versão do modelo
```

Essa arquitetura mantém o conceito central do projeto, mas adiciona os componentes necessários para operar o modelo em um ambiente distribuído.

---

## 21.3. Fluxo de dados

O fluxo produtivo poderia ocorrer em duas frentes: **inferência online** e **aprendizado contínuo**.

### Inferência online

```text
Cliente
   │
   ▼
Sistema consumidor
   │
   ▼
API Gateway
   │
   ▼
Serviço de inferência
   │
   ├── recebe contexto
   │
   ├── carrega features
   │
   ├── executa Thompson Sampling
   │
   └── seleciona ação
   │
   ▼
cellular / telephone
```

### Feedback e aprendizado

```text
Ação realizada
      │
      ▼
Resultado da campanha
      │
      ▼
Reward
      │
      ▼
Kinesis
      │
      ▼
S3
      │
      ▼
Processamento
      │
      ▼
Atualização / novo treinamento
      │
      ▼
Nova versão do modelo
```

Essa separação permite que a inferência continue disponível mesmo quando processos de treinamento ou atualização estiverem sendo executados.

---

## 21.4. Componentes AWS

### Amazon S3

O **Amazon S3** poderia atuar como camada de armazenamento do Data Lake.

Uma possível organização seria:

```text
s3://adaptive-marketing-platform/

├── raw/
│   └── bank-marketing/
│
├── processed/
│   └── features/
│
├── training/
│   ├── train/
│   └── test/
│
├── models/
│   ├── bandit/
│   └── encoder/
│
└── logs/
```

O S3 permitiria manter os dados históricos, datasets processados e artefatos necessários para reprodução dos experimentos.

---

### AWS Glue

O **AWS Glue** poderia ser utilizado para:

* catalogar os dados;
* executar processos de ETL;
* preparar os datasets;
* disponibilizar metadados;
* automatizar etapas de preparação.

O Glue Data Catalog poderia centralizar os metadados das tabelas utilizadas pelo pipeline analítico.

---

### Amazon Athena

O **Amazon Athena** poderia ser utilizado para consultas SQL diretamente sobre os dados armazenados no S3.

Isso permitiria realizar análises exploratórias, auditorias e consultas de acompanhamento sem necessidade de manter um banco dedicado para todos os dados históricos.

---

### Amazon Kinesis

O **Amazon Kinesis** poderia ser utilizado para receber os eventos gerados pelo sistema.

Exemplos:

```text
recommendation_generated
campaign_sent
campaign_opened
conversion
no_conversion
```

Cada evento poderia conter informações como:

```json
{
    "customer_id": "12345",
    "action": "cellular",
    "timestamp": "2026-08-08T14:00:00",
    "reward": 1
}
```

Os eventos poderiam ser direcionados posteriormente para armazenamento e processamento.

---

### Serviço de inferência

A API FastAPI atual poderia ser containerizada e executada em:

* **Amazon ECS com Fargate**, para uma API persistente e escalável; ou
* **AWS Lambda**, caso o volume e as características da inferência sejam compatíveis com um modelo serverless.

Para uma API de recomendação que precise manter maior controle sobre dependências, recursos e escalabilidade, **ECS/Fargate** seria uma alternativa adequada.

O fluxo seria:

```text
Cliente
   │
   ▼
API Gateway
   │
   ▼
ECS / Fargate
   │
   ▼
FastAPI
   │
   ▼
Contextual Thompson Sampling
   │
   ▼
Recomendação
```

---

### Amazon ECR

O **Amazon Elastic Container Registry (ECR)** poderia armazenar a imagem Docker da aplicação.

Uma possível esteira seria:

```text
Código
  │
  ▼
Build Docker
  │
  ▼
Amazon ECR
  │
  ▼
ECS / Fargate
```

Isso permitiria versionar as imagens da API e realizar deploys controlados.

---

### MLflow / Model Registry

O MLflow poderia continuar sendo utilizado para experiment tracking e gerenciamento das versões dos modelos.

Em uma arquitetura produtiva, os artefatos poderiam ser armazenados em S3 e o registry poderia controlar qual versão está aprovada para produção.

Exemplo:

```text
Model Registry

contextual-thompson-sampling
│
├── Version 1 → Archived
├── Version 2 → Staging
└── Version 3 → Production
```

Dessa forma, a aplicação poderia utilizar explicitamente uma versão aprovada do modelo.

---

## 21.5. Treinamento e atualização do modelo

O treinamento offline poderia ser transformado em um pipeline automatizado.

Uma possibilidade seria:

```text
Novos dados
    │
    ▼
Amazon S3
    │
    ▼
AWS Glue
    │
    ▼
Dataset de treinamento
    │
    ▼
Pipeline de treinamento
    │
    ▼
Thompson Sampling
    │
    ▼
Avaliação
    │
    ▼
MLflow
    │
    ▼
Model Registry
    │
    ▼
Aprovação
    │
    ▼
Deploy
```

O pipeline poderia ser executado:

* periodicamente;
* após determinado volume de novos dados;
* após mudança significativa na distribuição dos dados;
* ou conforme uma estratégia de atualização definida pelo negócio.

Antes de substituir o modelo atualmente utilizado, a nova versão deveria ser avaliada em relação ao baseline e à versão em produção.

---

## 21.6. Inferência online

Em produção, cada requisição poderia conter o contexto necessário para gerar a recomendação.

Exemplo:

```json
{
    "age": 35,
    "job": "admin.",
    "marital": "married",
    "education": "university.degree",
    "housing": "yes",
    "loan": "no",
    "month": "may",
    "day_of_week": "mon",
    "campaign": 1,
    "previous": 0,
    "poutcome": "nonexistent"
}
```

O serviço de inferência:

1. recebe o contexto;
2. aplica o mesmo preprocessing utilizado no treinamento;
3. transforma as variáveis categóricas utilizando o encoder versionado;
4. gera o vetor contextual;
5. executa o Thompson Sampling;
6. seleciona uma ação;
7. registra a decisão;
8. retorna a recomendação.

A resposta poderia seguir o formato:

```json
{
    "action": "cellular",
    "score": 0.8955,
    "model_version": "3"
}
```

A inclusão da versão do modelo facilita auditoria e rastreabilidade das decisões.

---

## 21.7. Atualização online do Bandit

Uma evolução importante em relação à implementação atual seria permitir que o modelo receba o reward depois que a ação for executada.

O fluxo seria:

```text
             Recomendação
                  │
                  ▼
          Ação executada
                  │
                  ▼
          Resultado observado
                  │
                  ▼
               Reward
                  │
                  ▼
             Event Stream
                  │
                  ▼
            Bandit Update
                  │
                  ▼
       Parâmetros atualizados
```

Por exemplo:

```text
action = cellular
reward = 1
```

indicaria que a ação escolhida resultou em conversão.

Já:

```text
action = cellular
reward = 0
```

representaria uma não conversão.

Esse mecanismo permitiria que o algoritmo aprendesse progressivamente com novas interações.

Em um ambiente produtivo, a atualização online deve ser projetada cuidadosamente para evitar corrupção do modelo, atualizações concorrentes e alterações não controladas da política.

Uma alternativa seria manter as atualizações em uma camada controlada e gerar novas versões do modelo em intervalos definidos.

---

## 21.8. Monitoramento e observabilidade

O ambiente produtivo deve monitorar não apenas a disponibilidade da API, mas também o comportamento do modelo.

### Monitoramento da aplicação

Métricas importantes:

* latência;
* taxa de erro;
* quantidade de requisições;
* disponibilidade;
* consumo de CPU;
* consumo de memória.

Essas métricas poderiam ser acompanhadas pelo **Amazon CloudWatch**.

### Monitoramento do modelo

Também deveriam ser acompanhadas métricas como:

* reward médio;
* taxa de conversão;
* distribuição das ações;
* quantidade de `cellular`;
* quantidade de `telephone`;
* reward por canal;
* reward por segmento;
* evolução temporal do reward.

Exemplo:

```text
Reward médio
     │
     │             ╭──────
     │         ╭───╯
     │     ╭───╯
     │  ╭──╯
     └──────────────────────
              tempo
```

### Data Drift

A distribuição das features também deveria ser monitorada.

Por exemplo:

```text
Distribuição no treinamento
          vs.
Distribuição em produção
```

Alterações significativas poderiam indicar **data drift**.

### Model Performance

Como o reward chega posteriormente à recomendação, seria possível calcular a performance real do modelo após o resultado da campanha ser observado.

Isso permitiria acompanhar:

```text
Performance histórica
        │
        ├── Baseline
        ├── Modelo atual
        └── Nova versão
```

---

## 21.9. Segurança e governança

Por se tratar de uma aplicação em contexto financeiro, a arquitetura deve considerar segurança e governança desde o início.

Entre as medidas possíveis:

* controle de acesso utilizando **IAM**;
* criptografia dos dados armazenados no S3;
* criptografia das comunicações;
* segregação entre ambientes de desenvolvimento, homologação e produção;
* controle de acesso aos modelos;
* versionamento dos artefatos;
* logs de decisões;
* rastreabilidade da versão do modelo utilizada;
* controle de alterações;
* políticas de retenção de dados;
* princípio do menor privilégio.

Cada recomendação deveria poder ser rastreada, quando permitido pelos requisitos de governança, até:

```text
Cliente/contexto
      │
      ▼
Versão do preprocessing
      │
      ▼
Versão do modelo
      │
      ▼
Ação escolhida
      │
      ▼
Reward observado
```

Isso facilita auditoria, análise de incidentes e avaliação das decisões do sistema.

---

## 21.10. Escalabilidade

A arquitetura proposta permite separar a escala da API da escala dos processos de treinamento.

Por exemplo:

```text
                    ┌──────────────────┐
                    │  API Gateway     │
                    └────────┬─────────┘
                             │
                  ┌──────────┴──────────┐
                  ▼                     ▼
            Instância 1            Instância 2
                  │                     │
                  └──────────┬──────────┘
                             ▼
                     Modelo Bandit
```

Com isso, várias instâncias da API poderiam atender requisições simultaneamente.

O processamento de dados e o treinamento poderiam ocorrer de forma independente:

```text
                API / Inferência
                       │
                       │
                       ▼
                 Recomendação


Dados ──► S3 ──► ETL ──► Treinamento
                           │
                           ▼
                       Novo modelo
```

Essa separação reduz o acoplamento entre inferência e treinamento e permite que cada componente seja escalado de acordo com sua necessidade.

---

## 21.11. Arquitetura resumida

A arquitetura produtiva poderia ser resumida da seguinte forma:

```text
                              ┌───────────────────┐
                              │ Sistemas Banco    │
                              └─────────┬─────────┘
                                        │
                                        ▼
                              ┌───────────────────┐
                              │   API Gateway     │
                              └─────────┬─────────┘
                                        │
                                        ▼
                              ┌───────────────────┐
                              │ ECS / Fargate     │
                              │ FastAPI           │
                              └─────────┬─────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
                 ┌───────────────┐             ┌───────────────┐
                 │ Modelo Bandit │             │ Features      │
                 │ versionado    │             │ / Dados       │
                 └───────┬───────┘             └───────────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Recomendação     │
                │ cellular /       │
                │ telephone        │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Resultado /      │
                │ Reward           │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Amazon Kinesis   │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Amazon S3        │
                │ Data Lake        │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ AWS Glue / ETL   │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Treinamento      │
                │ / atualização    │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ MLflow / Model   │
                │ Registry         │
                └────────┬─────────┘
                         │
                         ▼
                  Nova versão
                  do modelo
```

### Visão dos principais serviços

| Necessidade              | Serviço/tecnologia              |
| ------------------------ | ------------------------------- |
| API de inferência        | FastAPI                         |
| Exposição da API         | Amazon API Gateway              |
| Execução da API          | Amazon ECS / Fargate            |
| Container Registry       | Amazon ECR                      |
| Armazenamento            | Amazon S3                       |
| ETL e catálogo           | AWS Glue                        |
| Consulta dos dados       | Amazon Athena                   |
| Eventos e feedback       | Amazon Kinesis                  |
| Monitoramento            | Amazon CloudWatch               |
| Controle de acesso       | AWS IAM                         |
| Tracking de experimentos | MLflow                          |
| Registro de modelos      | MLflow Model Registry           |
| Treinamento              | Pipeline de ML / computação AWS |

A arquitetura apresentada é uma **proposta de evolução** da solução atual. Ela não foi implementada neste projeto e serve como desenho de referência para uma possível produtização.


---

# 23. Conclusão

O projeto demonstra uma solução end-to-end para seleção adaptativa de canais utilizando **Contextual Thompson Sampling**.

O fluxo desenvolvido parte da análise e preparação da base, passa pela construção de um baseline, implementação do algoritmo adaptativo, avaliação offline e chega à disponibilização do modelo para inferência.

No experimento realizado, o Thompson Sampling apresentou **reward médio de 0,1441**, contra **0,1212 do baseline**, representando um ganho relativo de aproximadamente **18,93%**.

Esse resultado indica que, dentro das limitações da avaliação offline utilizada, a política contextual conseguiu encontrar uma estratégia com maior recompensa média do que a política de referência.

O projeto também incorpora práticas de Machine Learning Engineering, como:

* organização modular do código;
* separação entre treinamento e inferência;
* persistência de modelos;
* persistência do encoder;
* API para servir recomendações;
* experimentação com notebooks;
* acompanhamento de experimentos com MLflow;
* documentação do projeto e das limitações.

Como evolução natural, a solução pode ser transformada em uma plataforma produtiva baseada em AWS, incorporando armazenamento centralizado, processamento de eventos, inferência escalável, monitoramento, governança e aprendizado contínuo.

A arquitetura proposta permite que o Contextual Thompson Sampling deixe de ser apenas um experimento offline e passe a funcionar como um componente de decisão adaptativa integrado ao ciclo real de campanhas, recebendo novos contextos, tomando decisões, observando rewards e evoluindo continuamente a política de seleção de canais.
