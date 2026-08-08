# Datathon

## Desafio

### O Datathon

O Datathon propõe um desafio prático no domínio financeiro: projetar uma plataforma de experimentação adaptativa para ofertas, mensagens ou próximos passos em canais digitais. Cada grupo constrói uma solução end-to-end de Machine Learning Engineering e demonstra como ela seria operada com observabilidade, avaliação e governança.

O objetivo não é reproduzir um sistema bancário real, mas sim mostrar maturidade técnica baseada nos conhecimentos do curso: formular o problema, construir baselines, versionar dados, servir componentes, avaliar qualidade, monitorar risco, documentar limitações e explicar decisões para públicos técnicos e de negócio, considerando o seguinte caso:

Uma instituição financeira digital precisa decidir, em diferentes canais, qual oferta, mensagem ou próximo passo apresentar para cada cliente elegível. Regras fixas e testes A/B longos desperdiçam tráfego, demoram para reagir a mudanças de contexto e dificultam a personalização responsável.

Esse é o ponto central de uma abordagem adaptativa (como multi-armed bandit): identificar comportamentos distintos, equilibrar exploração e explotação e aprender com respostas observadas sem congelar a decisão em regras estáticas.

---

## Referências algorítmicas

| Algoritmo                   | Papel no desafio                                                                                    | Evidência esperada                                                                             |
| --------------------------- | --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **Thompson Sampling**       | Exploração bayesiana sob incerteza para modelar conversão, clique ou recompensa esperada por braço. | Priors documentados, comparação com baseline e análise de exploração.                          |
| **Epsilon-Greedy ou UCB**   | Família de algoritmos para selecionar ações com base em recompensa esperada e exploração.           | Implementação ou adaptação justificada, com análise do trade-off entre exploração e conversão. |
| **Baseline determinístico** | Política simples de controle (regra fixa, melhor braço histórico ou segmentação inicial).           | Métrica comparativa clara para mostrar ganho ou limitação da política adaptativa.              |

O grupo pode implementar Thompson Sampling, Epsilon-Greedy, UCB ou outra variação contextual, desde que explique a escolha, mostre como o contexto entra na decisão e documente a estratégia de exploração e recompensas.

---

## Dados, regras e bases Kaggle

Use uma base Kaggle compatível com marketing, ofertas, propensão, campanhas, recomendação ou conversão como referência factual.

Não use dados reais de clientes, identificadores, patrimônio, renda, gênero, raça ou regras comerciais privadas.

Mantenha decisões sensíveis com humano no loop e documente base legal, finalidade, minimização e retenção.

### Bases Kaggle

| Base Kaggle                                    | Como usar no desafio                                                         |
| ---------------------------------------------- | ---------------------------------------------------------------------------- |
| **bank-marketing (henriqueyamahata)**          | Campanhas bancárias, propensão de conversão e decisão de oferta.             |
| **bank-marketing-data-set (tunguz)**           | Variação do problema de marketing bancário para comparação.                  |
| **bank-term-deposit-subscription (dharmik34)** | Assinatura de depósito a prazo como proxy de conversão.                      |
| **telemarketing-jyb-dataset (aguado)**         | Campanhas de contato e resposta, útil para comparação de canal ou abordagem. |

Outras bases serão aceitas se o grupo justificar a aderência e documentar fonte, versão, licença, colunas, target e limitações.

Descarte colunas de vazamento temporal (ex.: `duration` no Bank Marketing) e preserve a referência ao Kaggle.

---

# Entregáveis obrigatórios

Os entregáveis a seguir são organizados em nove etapas. O foco é a simplicidade e o funcionamento do pipeline básico.

Toda a documentação deve ser consolidada diretamente no arquivo `README.md` do repositório, sem necessidade de múltiplos arquivos soltos de governança.

---

## Etapa 0 — Organização do projeto

* Repositório público (ex: `datathon-7mlet-grupo-XX`).
* Arquivo `README.md` contendo a visão do problema e instruções de execução.
* Arquivo `requirements.txt` ou `pyproject.toml` com as dependências.

---

## Etapa 1 — Base Kaggle e EDA

* No `README.md`, insira o link da base Kaggle escolhida.
* Um notebook simples contendo a Análise Exploratória (EDA) e o tratamento de dados.

---

## Etapa 2 — Preparação da Base

* Se a base Kaggle escolhida já possuir dados claros de conversão/clique, você pode usá-la diretamente, sem precisar gerar dados sintéticos complexos.
* O objetivo é apenas ter as features do cliente e a variável alvo prontas para o modelo.

---

## Etapa 3 — Baseline e estratégia algorítmica

* No notebook, calcule a métrica de conversão de uma regra fixa (Baseline - ex: oferecer sempre o mesmo produto ou melhor histórico).
* Implemente o algoritmo adaptativo (Thompson Sampling ou Epsilon Greedy) e mostre a métrica dele superando o Baseline.

---

## Etapa 4 — Avaliação e Casos de Teste

* Cálculo das métricas de avaliação do modelo.
* No próprio notebook ou no `README.md`, crie um pequeno conjunto de testes com apenas 5 exemplos de clientes (Golden Set simplificado), mostrando qual oferta o modelo recomendou para cada um e se a decisão fez sentido.

---

## Etapa 5 — Serviço ou interface demonstrável

* Um script Python, um Notebook interativo ou uma API básica (FastAPI) que, ao receber os dados de um cliente, retorne a oferta recomendada.

---

## Etapa 6 — Arquitetura-alvo em Nuvem (AWS, Azure, Oracle, GCP)

* Escreva um ou dois parágrafos no `README.md` explicando, de forma simples, quais serviços da nuvem o grupo utilizaria para colocar esse projeto no ar.
* A criação de diagramas é opcional.

---

## Etapa 7 — Ciclo de vida MLOps

* Utilize a ferramenta de Controle de Versão para ML abordada no curso (ex: MLflow localmente) para registrar os parâmetros do seu modelo e as métricas obtidas na Etapa 3.

---

## Etapa 8 — Apresentação Final (Demo Day)

* Vídeo Pitch de até 5 minutos:

  * Explique rapidamente o problema de negócio.
  * Qual modelo foi usado.
  * Mostre a Etapa 5 rodando na prática (o modelo gerando uma recomendação).
* Não é necessário criar dezenas de slides.

---

# Critérios de avaliação

A avaliação segue o contrato da Fase 05 e valoriza o esforço de entrega do ciclo de ponta a ponta:

| Dimensão                     |    Peso | Critérios                                                                                                             | O que a banca procura                                                                                                 |
| ---------------------------- | ------: | --------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **Critérios de negócio**     | **30%** | Clareza na explicação do problema e impacto da solução.                                                               | Clareza na explicação do problema e impacto da solução.                                                               |
| **Validação técnica global** | **70%** | Código organizado, modelo funcionando e superando o baseline, uso básico de MLflow e sucesso na demonstração prática. | Código organizado, modelo funcionando e superando o baseline, uso básico de MLflow e sucesso na demonstração prática. |

---

# Checklist antes do Demo Day

* [ ] Repositório organizado com código e dependências (`requirements.txt` ou `pyproject.toml`).
* [ ] Notebook de EDA com a base Kaggle devidamente limpa e referenciada.
* [ ] Modelo Baseline e Modelo Adaptativo implementados e comparados.
* [ ] Notebook ou README demonstrando 5 casos de teste com as recomendações geradas.
* [ ] Código executável (script, notebook ou API) que retorna a predição funcionando perfeitamente.
* [ ] `README.md` preenchido com:

  * link da base;
  * parágrafo explicativo sobre a infraestrutura AWS;
  * instruções claras de execução local.
* [ ] Tracking de experimentos registrado via ferramenta MLOps (MLflow ou equivalente).
* [ ] Vídeo de apresentação (até 5 min) gravado demonstrando o código funcionando e justificando as escolhas.

---

# BOA SORTE!

**Datathon**
