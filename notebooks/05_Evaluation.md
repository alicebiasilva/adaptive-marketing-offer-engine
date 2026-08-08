# Avaliação do Baseline e Contextual Thompson Sampling

## 1. Objetivo

Esta etapa do projeto tem como objetivo estabelecer um **baseline de referência** e, posteriormente, implementar um **Contextual Thompson Sampling** para seleção do canal de contato mais adequado para cada cliente.

O problema é tratado como um problema de **Contextual Multi-Armed Bandit**, no qual:

* cada cliente representa um contexto;
* cada canal de contato representa uma ação (`arm`);
* a conversão (`y`) representa o reward;
* o modelo deve aprender a selecionar o canal com maior potencial de conversão para cada contexto.

Os resultados do Thompson Sampling são comparados com o baseline para verificar se uma política contextual consegue apresentar maior reward médio.

---

# 2. Notebook `03_Baseline.ipynb`

## 2.1 Objetivo

O notebook `03_Baseline.ipynb` estabelece uma referência simples para avaliar posteriormente o desempenho do Thompson Sampling.

O baseline não utiliza aprendizado por reforço nem otimização de uma política. Ele utiliza uma regra determinística baseada no histórico:

> Para cada perfil de cliente, selecionar o canal de contato mais frequente no conjunto de treinamento.

Dessa forma, o baseline representa uma estratégia simples baseada no comportamento histórico observado.

---

## 2.2 Variáveis utilizadas para definir o perfil

O perfil utilizado pelo baseline foi definido pelas seguintes características:

```text
job
education
marital
month
previous_contact
previous_success
age_group
financial_risk
engagement_score
```

Essas variáveis representam características demográficas, comportamentais, históricas e de contexto do cliente.

---

## 2.3 Construção do baseline

Primeiramente, os dados foram divididos em treinamento e teste utilizando divisão estratificada:

```text
train_test_split(
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

O conjunto de treinamento foi utilizado para identificar o canal majoritário para cada combinação de características do perfil.

Para cada perfil:

```text
(job,
 education,
 marital,
 month,
 previous_contact,
 previous_success,
 age_group,
 financial_risk,
 engagement_score)
```

foi identificado o canal (`cellular` ou `telephone`) mais frequente.

Quando um perfil não era encontrado no treinamento, foi utilizada a ação globalmente mais frequente como fallback.

---

## 2.4 Avaliação do baseline

O baseline foi aplicado ao conjunto de teste e avaliado utilizando métricas tradicionais de classificação:

* Accuracy
* Precision
* Recall
* F1-score
* ROC AUC

Resultado obtido:

| Métrica   |  Resultado |
| --------- | ---------: |
| Accuracy  | **0,8795** |
| Precision | **0,3502** |
| Recall    | **0,0819** |
| F1-score  | **0,1328** |
| ROC AUC   | **0,5313** |

### Interpretação

A Accuracy de aproximadamente 87,95% deve ser interpretada com cautela devido ao desbalanceamento da variável `y`.

O Recall de apenas 8,19% indica que o baseline identifica uma parcela pequena dos clientes que efetivamente convertem.

O F1-score de 0,1328 também indica baixo desempenho na identificação da classe positiva.

A ROC AUC de 0,5313 está apenas ligeiramente acima de 0,5, indicando baixa capacidade de discriminação.

Portanto, embora o baseline apresente Accuracy relativamente alta, ele possui baixa capacidade de identificar conversões.

---

## 2.5 Análise das características

Foi realizada uma análise de remoção de features para verificar o impacto das variáveis utilizadas no baseline.

O resultado indicou que `engagement_score`, `financial_risk`, `education`, `month` e `job` apresentavam maior impacto sobre o F1-score do baseline quando comparadas às demais variáveis.

Essa análise serviu como referência para a construção da política contextual posteriormente utilizada no Thompson Sampling.

---

# 3. Notebook `04_ThompsonSampling.ipynb`

## 3.1 Objetivo

O notebook `04_ThompsonSampling.ipynb` implementa uma primeira versão experimental de **Contextual Thompson Sampling**.

Diferentemente do baseline, o objetivo não é prever diretamente a variável `y`.

O objetivo é:

> **selecionar, para cada cliente, o canal de contato que apresenta maior potencial de reward considerando suas características.**

---

# 4. Definição do problema como Contextual Bandit

O problema foi estruturado da seguinte forma:

| Elemento | Definição                    |
| -------- | ---------------------------- |
| Contexto | Características do cliente   |
| Arms     | `cellular` e `telephone`     |
| Ação     | Canal de contato escolhido   |
| Reward   | Conversão (`y`)              |
| Política | Contextual Thompson Sampling |

As ações utilizadas foram:

```text
cellular
telephone
```

---

# 5. Dados utilizados

O dataset contém características do cliente, informações históricas de contato, variáveis econômicas e a variável alvo `y`.

Entre as características utilizadas como contexto estão:

```text
age
job
marital
education
default
housing
loan
contact
month
day_of_week
duration
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

O canal `contact` foi tratado como a **ação histórica observada**, e não como uma feature utilizada diretamente para decidir a ação do Thompson.

---

# 6. Preparação do contexto

As variáveis categóricas foram transformadas utilizando **One-Hot Encoding**, enquanto as variáveis numéricas foram mantidas em formato numérico e, quando necessário para o modelo linear, preparadas para compor o vetor de contexto.

Após o processamento, o contexto utilizado pelo modelo apresentou:

```text
Shape do contexto de treino: (32950, 69)
Shape do contexto de teste:  (8238, 69)

Shape final do contexto de treino: (32950, 70)
Shape final do contexto de teste:  (8238, 70)
```

O vetor final de contexto possui:

```text
70 dimensões
```

---

# 7. Estrutura do modelo

Foi utilizada uma abordagem de **Contextual Linear Thompson Sampling**.

Para cada arm (`cellular` e `telephone`) foram mantidos parâmetros específicos:

```text
A
b
```

com dimensões:

```text
A = (70, 70)
b = (70,)
```

A estrutura permite estimar uma relação entre as características do cliente e o reward esperado para cada canal.

O modelo mantém uma distribuição de incerteza sobre os parâmetros e realiza amostragens dessa distribuição para tomar decisões.

Essa característica diferencia Thompson Sampling de um modelo supervisionado tradicional.

---

# 8. Processo de escolha da ação

Para cada cliente:

1. O vetor de contexto é recebido.
2. O modelo calcula os parâmetros estimados para cada arm.
3. Uma amostra é obtida considerando a incerteza dos parâmetros.
4. Cada canal recebe um score.
5. O canal com maior score é selecionado.
6. O processo é repetido para os demais clientes.

Assim, a política pode escolher diferentes canais dependendo do contexto.

---

# 9. Distribuição das ações escolhidas

No conjunto de teste, a política produziu:

| Canal       | Quantidade | Percentual |
| ----------- | ---------: | ---------: |
| `cellular`  |      5.889 | **71,49%** |
| `telephone` |      2.349 | **28,51%** |
| **Total**   |  **8.238** |   **100%** |

Para comparação, a distribuição histórica das ações foi:

| Canal       | Quantidade | Percentual |
| ----------- | ---------: | ---------: |
| `cellular`  |      5.236 | **63,56%** |
| `telephone` |      3.002 | **36,44%** |

Portanto, o Thompson Sampling não simplesmente reproduziu a distribuição histórica. Ele aumentou a proporção de recomendações de `cellular` de aproximadamente 63,56% para 71,49%.

---

# 10. Análise do comportamento contextual

Foi analisada a distribuição das ações escolhidas pelo modelo condicionada às características dos clientes.

Essa análise demonstrou que o comportamento do modelo não é uniforme para todos os clientes.

## 10.1 `month`

A variável `month` apresentou uma das maiores diferenciações.

| Mês |   Cellular |  Telephone |
| --- | ---------: | ---------: |
| apr |     32,36% | **67,64%** |
| aug |     70,08% |     29,92% |
| dec |     46,15% | **53,85%** |
| jul |     68,82% |     31,18% |
| jun | **92,53%** |      7,47% |
| mar |     78,18% |     21,82% |
| may |     81,76% |     18,24% |
| nov |     46,95% | **53,05%** |
| oct |     42,75% | **57,25%** |
| sep |     76,42% |     23,58% |

A política chega a inverter a preferência entre os canais dependendo do mês.

Esse comportamento fornece evidência de que o modelo está utilizando o contexto para modificar sua decisão.

---

## 10.2 `previous_success`

| previous_success | Cellular |  Telephone |
| ---------------- | -------: | ---------: |
| 0                |   72,07% |     27,93% |
| 1                |   54,10% | **45,90%** |

Clientes com histórico de sucesso apresentaram uma maior proporção de recomendações de `telephone`.

---

## 10.3 `previous_contact`

| previous_contact | Cellular |  Telephone |
| ---------------- | -------: | ---------: |
| 0                |   74,18% |     25,82% |
| 1                |   53,80% | **46,20%** |

O histórico de contato também apresentou forte diferenciação na política.

---

## 10.4 `job`

A variável `job` apresentou influência moderada.

Por exemplo:

```text
management      → 25,39% telephone
housemaid       → 37,56% telephone
unknown         → 38,46% telephone
```

Isso indica que diferentes perfis ocupacionais recebem políticas diferentes.

---

## 10.5 Features com menor diferenciação

Variáveis como:

```text
financial_risk
education
age_group
marital
```

apresentaram menor variação na distribuição dos canais.

Por exemplo, para `financial_risk`:

| Financial risk | Cellular | Telephone |
| -------------- | -------: | --------: |
| 0              |   72,45% |    27,55% |
| 1              |   70,77% |    29,23% |
| 2              |   71,01% |    28,99% |

Portanto, nessa primeira versão, essa variável apresentou pouca influência sobre a escolha entre os dois canais.

---

# 11. Avaliação offline

Como o projeto utiliza dados históricos, a primeira avaliação do Thompson Sampling foi realizada offline.

Para cada cliente existem:

```text
contexto
+
ação histórica
+
reward observado
```

O Thompson Sampling produz uma ação recomendada.

A ação recomendada pode:

```text
coincidir com a ação histórica
```

ou:

```text
ser diferente da ação histórica
```

Quando há coincidência, o reward histórico pode ser utilizado diretamente na avaliação.

Esse método é conhecido como **replay/off-policy evaluation**.

---

# 12. Matriz de coincidência do Thompson Sampling

A comparação entre a ação histórica e a ação escolhida pelo Thompson foi:

| Histórico \ Thompson |  cellular | telephone |     Total |
| -------------------- | --------: | --------: | --------: |
| cellular             |     3.359 |     1.877 |     5.236 |
| telephone            |     2.530 |       472 |     3.002 |
| **Total**            | **5.889** | **2.349** | **8.238** |

As ações coincidentes correspondem à diagonal:

```text
cellular → cellular = 3.359
telephone → telephone = 472
```

Total:

```text
3.359 + 472 = 3.831
```

Taxa de coincidência:

```text
3.831 / 8.238 = 46,50%
```

Portanto:

> O Thompson Sampling coincidiu com a ação histórica em aproximadamente **46,50%** das observações.

---

# 13. Comparação final

A avaliação offline apresentou os seguintes resultados:

| Modelo                | Observações avaliadas | Taxa de coincidência | Reward médio | Conversões observadas |
| --------------------- | --------------------: | -------------------: | -----------: | --------------------: |
| **Baseline**          |                 6.240 |               75,75% |       12,12% |                   756 |
| **Thompson Sampling** |                 3.831 |               46,50% |   **14,41%** |                   522 |

O reward médio foi:

```text
Baseline           = 0,1212
Thompson Sampling  = 0,1441
```

Diferença absoluta:

```text
0,1441 - 0,1212 = 0,0229
```

Portanto, o Thompson apresentou um ganho absoluto de:

```text
2,29 pontos percentuais
```

O ganho relativo foi calculado como:

```text
((0,1441 / 0,1212) - 1) × 100
```

resultando em aproximadamente:

```text
18,93%
```

---

# 14. Interpretação dos resultados

Na avaliação offline realizada, o Contextual Thompson Sampling apresentou **reward médio de 14,41%**, superior aos **12,12%** observados para o baseline.

Isso representa:

* **+2,29 pontos percentuais** de ganho absoluto;
* **+18,93%** de ganho relativo de reward.

Apesar de apresentar uma taxa de coincidência menor que a do baseline, o Thompson Sampling apresentou maior reward médio nas observações em que sua recomendação coincidiu com a ação historicamente realizada.

Esse resultado sugere que a política contextual pode estar identificando combinações de características e canais associadas a maior probabilidade de conversão.

---

# 15. Limitações da avaliação

A comparação deve ser interpretada como uma **avaliação offline preliminar**.

O dataset histórico informa apenas:

```text
Cliente
   ↓
Canal que foi utilizado
   ↓
Conversão observada
```

Ele não informa simultaneamente:

```text
Cliente
   ├── cellular → conversão
   └── telephone → conversão
```

Portanto, quando o Thompson recomenda uma ação diferente daquela utilizada historicamente, não é possível saber qual teria sido o reward contrafactual.

Por esse motivo, a avaliação por replay utiliza apenas as observações nas quais a ação recomendada coincide com a ação histórica.

Consequentemente:

> O ganho de 18,93% deve ser interpretado como um resultado de avaliação offline e não como uma estimativa causal definitiva de aumento de conversão em produção.

Para uma avaliação mais robusta, seria necessário utilizar dados com **propensões de logging**, uma política de exploração conhecida ou realizar uma validação online controlada.

---

# 16. Conclusão

O baseline estabeleceu uma referência simples baseada na escolha do canal mais frequente para cada perfil de cliente.

O Contextual Thompson Sampling introduziu uma política adaptativa capaz de considerar as características do cliente e modificar a escolha do canal de acordo com o contexto.

A análise da política mostrou que determinadas variáveis, especialmente:

```text
month
previous_success
previous_contact
job
```

produzem mudanças relevantes na escolha entre `cellular` e `telephone`.

Na avaliação offline, o Thompson Sampling apresentou:

```text
Reward médio: 14,41%
```

contra:

```text
Reward médio do baseline: 12,12%
```

resultando em:

```text
Ganho absoluto: +2,29 p.p.
Ganho relativo: +18,93%
```

Dessa forma, a primeira implementação do Contextual Thompson Sampling apresentou **potencial de melhoria em relação ao baseline**, justificando a continuidade do desenvolvimento do modelo e a realização de avaliações mais robustas.

A próxima etapa recomendada é validar a política utilizando uma metodologia de avaliação offline mais completa, como **Inverse Propensity Scoring (IPS)**, caso as propensões das ações históricas estejam disponíveis, e posteriormente realizar validação online para verificar o impacto real sobre a conversão.
