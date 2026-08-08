# Datathon — Plataforma Adaptativa de Ofertas com Contextual Thompson Sampling

## 1. Visão geral

Este projeto foi desenvolvido como solução para o **Datathon**, com o objetivo de construir uma plataforma de experimentação adaptativa capaz de selecionar o melhor canal de abordagem para cada cliente com base em seu contexto e no histórico de conversões.

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

A referência da base e sua documentação estão disponíveis no diretório:

```text
/docs
```

A documentação original do **Tech Challenge** também está disponível nesse diretório.

### Documentação

```text
docs/
├── DatabaseBankMarketing.md
└── TechChallenge.md
```

Esses arquivos devem ser consultados para obter detalhes sobre:

* origem da base;
* descrição das variáveis;
* regras do desafio;
* critérios de avaliação;
* entregáveis obrigatórios;
* restrições do projeto.

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
* possíveis fontes de vazamento de informação.

Um ponto importante identificado foi o **desbalanceamento da variável alvo `y`**, com predominância da classe de não conversão.

Por isso, métricas como **Accuracy** não são suficientes para avaliar o modelo. Foram consideradas também:

* Precision;
* Recall;
* F1-score;
* ROC AUC;
* taxa de conversão/reward.

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

Foram avaliadas características como:

* `job`;
* `education`;
* `marital`;
* `month`;
* `previous_contact`;
* `previous_success`;
* `age_group`;
* `financial_risk`;
* `engagement_score`.

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
│   ├── bank_marketing.md
│   └── datathon.md
│
├── models/
│   ├── bandit.pkl
│   └── encoder.pkl
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_FeatureEngineering.ipynb
│   ├── 03_Baseline.ipynb
│   └── 04_ThompsonSampling.ipynb
│
├── scripts/
│   └── train.py
│
├── src/
│   ├── bandit.py
│   ├── config.py
│   ├── inference.py
│   └── preprocessing.py
│
├── mlruns/
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
                 ┌─────────┴─────────┐
                 ▼                   ▼
           bandit.pkl          encoder.pkl
                 │                   │
                 └─────────┬─────────┘
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
git clone <URL_DO_REPOSITORIO>
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

# 21. Próximos passos

Como evolução do projeto, podem ser implementados:

* atualização online do modelo após cada reward observado;
* endpoint `/update` para alimentar o bandit com novos resultados;
* monitoramento de distribuição das ações;
* monitoramento de reward;
* monitoramento de drift;
* Golden Set com 5 clientes;
* registro completo da comparação com baseline no MLflow;
* arquitetura em nuvem;
* governança e monitoramento de decisões;
* testes automatizados;
* containerização da API.

---

# 22. Conclusão

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