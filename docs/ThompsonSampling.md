# Thompson Sampling — Documentação

## 1. Visão geral

O **Thompson Sampling** é um algoritmo de **Multi-Armed Bandit (MAB)**, pertencente ao aprendizado por reforço, utilizado para tomar decisões sequenciais em cenários nos quais existem diferentes ações possíveis e o objetivo é aprender, ao longo do tempo, qual ação gera melhores resultados.

Diferentemente de um modelo supervisionado tradicional, que aprende uma relação entre variáveis de entrada e uma variável-alvo a partir de um conjunto de dados histórico, um algoritmo Bandit precisa lidar simultaneamente com:

* **Exploração:** testar ações sobre as quais ainda existe pouca informação;
* **Explotação:** escolher ações que já demonstraram bom desempenho;
* **Feedback:** utilizar o resultado de cada decisão para atualizar o conhecimento sobre as ações.

No contexto deste projeto, o Thompson Sampling pode ser utilizado para decidir **qual oferta/canal deve ser apresentado a cada cliente**, utilizando a conversão como recompensa.

---

# 2. Problema que o Thompson Sampling resolve

Imagine que existam três possíveis ações:

```text
Ação 1 → Telefone
Ação 2 → E-mail
Ação 3 → Outro canal
```

O objetivo é descobrir qual ação possui maior probabilidade de gerar uma conversão.

Inicialmente, o algoritmo não sabe qual é a melhor ação.

Por exemplo:

```text
Telefone → ?
E-mail   → ?
Outro    → ?
```

Conforme clientes são abordados, o algoritmo observa os resultados:

```text
Telefone → converteu
E-mail   → não converteu
Telefone → não converteu
Outro    → converteu
...
```

Essas observações são utilizadas para atualizar a estimativa de desempenho de cada ação.

O objetivo não é simplesmente encontrar a ação com maior taxa histórica de conversão, mas **maximizar a quantidade acumulada de recompensas ao longo do tempo**.

---

# 3. Multi-Armed Bandit

O problema pode ser representado como uma máquina caça-níqueis com vários braços.

Cada braço representa uma ação:

```text
                 Multi-Armed Bandit

                  ┌──────────────┐
                  │   Cliente    │
                  └──────┬───────┘
                         │
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
      ┌────────┐     ┌────────┐     ┌────────┐
      │ Ação A │     │ Ação B │     │ Ação C │
      └────────┘     └────────┘     └────────┘
          │              │              │
          ↓              ↓              ↓
       reward         reward         reward
```

Cada ação possui uma probabilidade desconhecida de gerar recompensa.

O algoritmo precisa aprender essas probabilidades enquanto toma decisões.

---

# 4. Exploração vs. explotação

Esse é um dos principais conceitos de algoritmos Bandit.

## 4.1 Exploração

Explorar significa escolher ações sobre as quais o algoritmo ainda possui pouca informação.

Exemplo:

```text
E-mail:
20 tentativas
5 conversões

Telefone:
2 tentativas
1 conversão
```

Embora o telefone tenha uma taxa observada maior:

```text
Telefone = 50%
E-mail   = 25%
```

a quantidade de observações é muito menor.

O algoritmo deve continuar explorando o telefone para descobrir se essa taxa elevada é realmente consistente.

---

## 4.2 Explotação

Explotar significa utilizar a ação que atualmente parece apresentar melhor desempenho.

Por exemplo:

```text
Telefone → maior probabilidade estimada de conversão
```

O algoritmo pode escolher telefone com maior frequência.

---

## 4.3 O equilíbrio

Um bom algoritmo Bandit precisa equilibrar:

```text
Exploração
    ↓
Aprender sobre ações desconhecidas

        +

Explotação
    ↓
Utilizar ações que parecem melhores
```

O Thompson Sampling realiza esse equilíbrio de forma probabilística, sem precisar definir manualmente uma taxa fixa de exploração.

---

# 5. Intuição do Thompson Sampling

A ideia central é representar a incerteza sobre cada ação utilizando uma **distribuição de probabilidade**.

Para problemas de recompensa binária, como:

```text
Conversão = 1
Não conversão = 0
```

a distribuição Beta é particularmente adequada.

Para cada ação, mantemos:

```text
α (alpha)
β (beta)
```

Esses parâmetros representam nossa crença sobre a probabilidade de sucesso daquela ação.

A distribuição fica:

```text
θ ~ Beta(α, β)
```

onde `θ` representa uma possível taxa de conversão da ação.

---

# 6. Distribuição Beta

A distribuição Beta possui valores entre 0 e 1, sendo portanto adequada para representar probabilidades.

Exemplo:

```text
Beta(1, 1)
```

representa uma distribuição uniforme.

Inicialmente, não temos conhecimento suficiente para dizer que uma ação é melhor que outra.

Depois de algumas observações, podemos ter:

```text
Telefone → Beta(8, 4)
```

Isso significa que já observamos:

```text
7 conversões
3 não conversões
```

considerando uma inicialização:

```text
α = 1 + sucessos
β = 1 + fracassos
```

A distribuição passa então a concentrar maior probabilidade em valores mais altos de conversão.

---

# 7. Parâmetros do Thompson Sampling

Para cada ação, são mantidos dois parâmetros principais.

## Alpha

Representa os sucessos observados.

```text
α = α_inicial + número_de_sucessos
```

## Beta

Representa os fracassos observados.

```text
β = β_inicial + número_de_fracassos
```

Por exemplo:

```text
Ação: telefone

Sucessos = 10
Fracassos = 5

α = 1 + 10 = 11
β = 1 + 5  = 6
```

A distribuição utilizada será:

```text
Beta(11, 6)
```

---

# 8. Inicialização

Uma inicialização comum é:

```text
α = 1
β = 1
```

Para todas as ações.

Por exemplo:

```text
Telefone → Beta(1,1)
E-mail   → Beta(1,1)
SMS      → Beta(1,1)
```

Isso significa que todas começam com o mesmo nível de conhecimento.

Essa inicialização também é chamada de **prior uniforme**.

---

# 9. Funcionamento passo a passo

A cada interação com um cliente, o algoritmo executa aproximadamente o seguinte processo.

## Passo 1 — Obter as ações disponíveis

Exemplo:

```python
actions = [
    "telefone",
    "email",
    "sms"
]
```

---

## Passo 2 — Amostrar uma probabilidade

Para cada ação, o algoritmo sorteia um valor da distribuição Beta correspondente.

Exemplo:

```text
Telefone → Beta(8,4) → 0.71
E-mail   → Beta(5,5) → 0.43
SMS      → Beta(2,3) → 0.61
```

---

## Passo 3 — Selecionar a ação

O algoritmo escolhe a ação cujo valor amostrado foi maior.

Neste exemplo:

```text
Telefone = 0.71
E-mail   = 0.43
SMS      = 0.61
```

Portanto:

```text
Ação escolhida = Telefone
```

---

## Passo 4 — Observar a recompensa

Depois da interação, observa-se o resultado.

Por exemplo:

```text
Conversão = 1
```

ou:

```text
Conversão = 0
```

---

## Passo 5 — Atualizar a distribuição

Se houve conversão:

```text
α = α + 1
```

Se não houve conversão:

```text
β = β + 1
```

Por exemplo:

```text
Antes:

Telefone → α=8, β=4

Resultado:
Conversão = 1

Depois:

Telefone → α=9, β=4
```

O processo continua para o próximo cliente.

---

# 10. Por que a amostragem gera exploração?

Essa é uma das características mais importantes do Thompson Sampling.

Imagine:

```text
Ação A:
α = 90
β = 10

Ação B:
α = 2
β = 1
```

A ação A possui muito mais evidências de sucesso.

Entretanto, a ação B possui maior incerteza.

Ao realizar uma amostragem da distribuição Beta de cada ação, ocasionalmente a ação B poderá gerar uma amostra maior que A.

Isso faz com que B seja explorada.

Portanto:

> Quanto maior a incerteza sobre uma ação, maior a possibilidade de ela ser explorada.

Ao mesmo tempo, ações que consistentemente apresentam bons resultados tendem a gerar amostras maiores e passam a ser escolhidas com maior frequência.

---

# 11. Estrutura dos dados de entrada

O Thompson Sampling não precisa necessariamente de um dataset inteiro para funcionar.

Ele pode operar de forma **online**, recebendo uma observação por vez.

Uma interação pode ser representada por:

```text
cliente
ação escolhida
recompensa
```

Por exemplo:

| cliente | ação     | recompensa |
| ------- | -------- | ---------: |
| 1       | telefone |          1 |
| 2       | email    |          0 |
| 3       | telefone |          1 |
| 4       | sms      |          0 |
| 5       | email    |          1 |

A recompensa deve ser definida de acordo com o objetivo do problema.

---

# 12. Recompensa no projeto

Para o problema de conversão do Bank Marketing, uma representação simples é:

```text
Conversão = 1
Não conversão = 0
```

Portanto:

```python
reward = 1
```

quando o cliente aceita/converte, e:

```python
reward = 0
```

quando não converte.

Essa estrutura permite utilizar diretamente o modelo Beta-Bernoulli.

---

# 13. Contexto do cliente

Existe uma diferença importante entre um **Multi-Armed Bandit tradicional** e um **Contextual Bandit**.

No Bandit tradicional, a escolha depende apenas das informações acumuladas sobre as ações.

Exemplo:

```text
Qual canal apresenta maior conversão?
```

No Contextual Bandit, a decisão também considera características do cliente.

Exemplo:

```text
idade
job
education
marital
housing
loan
contact
month
poutcome
...
```

A decisão passa a ser:

```text
Dado este cliente,
qual ação possui maior probabilidade de gerar conversão?
```

Essa abordagem é mais adequada quando diferentes perfis de clientes respondem melhor a diferentes ofertas ou canais.

---

# 14. Dados históricos e treinamento

Uma característica importante do Thompson Sampling é que ele **não possui treinamento da mesma forma que modelos como Random Forest, XGBoost ou redes neurais**.

Não existe necessariamente uma etapa:

```text
fit(X_train, y_train)
```

como em aprendizado supervisionado tradicional.

O aprendizado acontece durante as interações.

O algoritmo mantém um estado:

```text
α_action
β_action
```

e atualiza esse estado conforme recebe recompensas.

Portanto, o "treinamento" consiste essencialmente em alimentar o algoritmo com interações históricas ou reais.

---

# 15. Uso de dados históricos

Em um projeto offline, é possível utilizar um dataset histórico para simular as interações.

Exemplo:

```text
Cliente 1 → ação = telefone → reward = 1
Cliente 2 → ação = email    → reward = 0
Cliente 3 → ação = telefone → reward = 0
...
```

O algoritmo percorre essas observações sequencialmente e atualiza seus parâmetros.

Entretanto, existe uma consideração importante:

> Os dados históricos foram gerados por uma política anterior.

Isso significa que talvez nem todas as ações tenham sido testadas para todos os clientes.

Por exemplo:

```text
Cliente A → histórico mostra telefone
```

mas não sabemos necessariamente:

```text
Cliente A → o que teria acontecido se tivesse recebido e-mail?
```

Esse problema é conhecido como **counterfactual problem**.

Por isso, avaliação offline de Bandits precisa ser feita com cuidado.

---

# 16. Estrutura recomendada para o projeto

Uma arquitetura conceitual pode ser:

```text
                 Dados do cliente
                       │
                       ↓
              ┌─────────────────┐
              │ Thompson Sampling│
              └────────┬────────┘
                       │
                       ↓
                Ação escolhida
                       │
                       ↓
                 Interação
                       │
                       ↓
                  Conversão?
                  /       \
                Sim       Não
                 │          │
                 ↓          ↓
              reward=1   reward=0
                 │          │
                 └────┬─────┘
                      ↓
                Atualização
               dos parâmetros
```

---

# 17. Entrada do algoritmo

A entrada pode conter dois tipos de informação.

## 17.1 Contexto

Características do cliente:

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
poutcome
...
```

## 17.2 Ações disponíveis

Exemplo:

```python
actions = [
    "telefone",
    "email",
    "sms"
]
```

## 17.3 Recompensa

Após a ação ser realizada:

```text
reward ∈ {0,1}
```

---

# 18. Saída esperada

A saída principal do Thompson Sampling é a **ação selecionada**.

Exemplo:

```python
{
    "customer_id": 123,
    "recommended_action": "telefone"
}
```

Em uma API, também pode ser interessante retornar informações adicionais:

```json
{
    "recommended_action": "telefone",
    "probability_sample": 0.73
}
```

Ou, para análise:

```json
{
    "recommended_action": "telefone",
    "scores": {
        "telefone": 0.73,
        "email": 0.41,
        "sms": 0.58
    }
}
```

Esses valores são amostras da distribuição, e **não devem ser interpretados automaticamente como probabilidades reais de conversão**.

---

# 19. Parâmetros do algoritmo

Uma implementação simples possui poucos hiperparâmetros.

## Número de ações

Quantidade de ações disponíveis:

```python
n_actions
```

Exemplo:

```text
3
```

---

## Alpha inicial

```python
alpha_init = 1
```

---

## Beta inicial

```python
beta_init = 1
```

---

## Random seed

Pode ser utilizado para reproduzir experimentos:

```python
random_state = 42
```

---

# 20. Estado interno

O algoritmo precisa armazenar o estado de cada ação.

Exemplo:

```python
alpha = {
    "telefone": 8,
    "email": 5,
    "sms": 2
}

beta = {
    "telefone": 4,
    "email": 6,
    "sms": 3
}
```

Também é possível armazenar:

```python
successes
failures
total_trials
```

para facilitar monitoramento e análise.

---

# 21. Métricas importantes

A avaliação de um Bandit não deve se limitar a accuracy, precision, recall e F1.

Algumas métricas mais apropriadas são:

## Taxa de conversão

```text
conversões / interações
```

---

## Reward acumulado

```text
R_total = Σ reward_t
```

Quanto maior, melhor.

---

## Reward médio

```text
R_médio = reward_total / número_de_interações
```

---

## Regret

O regret mede a diferença entre o resultado obtido e o resultado que seria obtido caso a melhor ação estivesse sempre disponível.

De forma simplificada:

```text
Regret =
reward da melhor ação
-
reward da ação escolhida
```

O objetivo de um bom algoritmo Bandit é minimizar o regret ao longo do tempo.

---

# 22. Curva de aprendizado

Uma análise interessante é acompanhar o reward acumulado:

```text
Interações
    │
    │                  /
    │                /
    │             __/
    │          __/
    │       __/
    │______/________________
```

Se o algoritmo estiver aprendendo adequadamente, espera-se que seu desempenho melhore à medida que novas interações são observadas.

Também pode ser interessante acompanhar:

```text
Taxa de conversão por ação
Número de vezes que cada ação foi escolhida
Reward acumulado
Regret
Distribuição dos parâmetros α e β
```

---

# 23. Thompson Sampling não é um classificador

Essa distinção é importante para este projeto.

Um classificador tradicional responde:

```text
Qual é a probabilidade de conversão deste cliente?
```

Por exemplo:

```text
P(y=1 | cliente) = 0.72
```

O Thompson Sampling responde:

```text
Qual ação devo escolher?
```

Por exemplo:

```text
ação recomendada = telefone
```

Portanto, eles podem ser utilizados de maneiras diferentes.

---

# 24. Thompson Sampling contextual

Para utilizar as características individuais do cliente diretamente na decisão, é necessário utilizar uma variante de **Contextual Thompson Sampling**.

Nesse caso, a decisão pode considerar:

```text
cliente
   ↓
features
   ↓
modelo contextual
   ↓
estimativa das ações
   ↓
amostragem
   ↓
ação
```

Exemplos de abordagens incluem:

* Bayesian Linear Thompson Sampling;
* Logistic Thompson Sampling;
* Neural Thompson Sampling.

Para um primeiro modelo, entretanto, o **Beta-Bernoulli Thompson Sampling** é significativamente mais simples de implementar e interpretar.

---

# 25. Como utilizar no Bank Marketing

No contexto do dataset Bank Marketing, uma implementação inicial pode definir:

### Contexto

```text
job
marital
education
default
housing
loan
contact
month
day_of_week
poutcome
```

### Ações

As ações podem representar diferentes canais/ofertas definidos no projeto.

Exemplo:

```text
telefone
email
sms
```

### Reward

```text
1 → cliente converteu
0 → cliente não converteu
```

O algoritmo então aprende quais ações apresentam maior retorno.

---

# 26. Exemplo conceitual

Inicialmente:

```text
Telefone → Beta(1,1)
E-mail   → Beta(1,1)
SMS      → Beta(1,1)
```

Cliente 1:

```text
Amostras:

Telefone = 0.40
E-mail   = 0.80
SMS      = 0.30

Escolha:
E-mail
```

Resultado:

```text
Não converteu
```

Atualização:

```text
E-mail → α=1, β=2
```

Cliente 2:

```text
Telefone = 0.70
E-mail   = 0.25
SMS      = 0.60

Escolha:
Telefone
```

Resultado:

```text
Converteu
```

Atualização:

```text
Telefone → α=2, β=1
```

O processo continua dessa maneira.

---

# 27. Pseudocódigo

```text
Inicializar α e β para cada ação

Para cada cliente:

    Para cada ação:
        amostrar θ ~ Beta(α, β)

    escolher ação com maior θ

    executar ação

    observar reward

    se reward == 1:
        α[ação] = α[ação] + 1
    senão:
        β[ação] = β[ação] + 1
```

Essa é a essência do algoritmo.

---

# 28. Diferença para ε-greedy

Outro algoritmo comum de Multi-Armed Bandit é o **ε-greedy**.

No ε-greedy:

```text
probabilidade ε → explorar
probabilidade 1-ε → explotar
```

Por exemplo:

```text
ε = 0.10
```

significa aproximadamente:

```text
10% → exploração
90% → explotação
```

No Thompson Sampling não é necessário definir explicitamente essa porcentagem.

A exploração emerge da incerteza das distribuições.

---

# 29. Diferença para UCB

O **Upper Confidence Bound (UCB)** também utiliza a incerteza das ações.

De forma simplificada:

```text
UCB =
média estimada
+
bônus de incerteza
```

O Thompson Sampling, por outro lado, utiliza amostragem das distribuições posteriores:

```text
θ ~ distribuição posterior
```

e escolhe a ação com maior amostra.

---

# 30. Vantagens

O Thompson Sampling possui algumas vantagens importantes:

* Implementação relativamente simples;
* Boa capacidade de equilibrar exploração e explotação;
* Considera a incerteza das estimativas;
* Não exige uma taxa de exploração fixa;
* Pode ser atualizado incrementalmente;
* É adequado para decisões sequenciais;
* Pode funcionar muito bem em problemas de recomendação e conversão.

---

# 31. Limitações

Também existem limitações.

### 1. Necessidade de feedback

O algoritmo precisa receber recompensas para aprender.

Sem feedback:

```text
ação → ?
```

não existe atualização.

---

### 2. Problema de dados históricos

Dados históricos podem ter sido produzidos por uma política anterior.

Isso dificulta saber o que teria acontecido com ações que não foram escolhidas.

---

### 3. Bandit simples não utiliza contexto

O Beta-Bernoulli Thompson Sampling básico mantém uma distribuição por ação.

Portanto:

```text
Telefone → uma distribuição
E-mail → uma distribuição
```

Ele não diferencia automaticamente:

```text
Cliente jovem
Cliente idoso
Cliente com empréstimo
Cliente sem empréstimo
```

Para isso, deve-se utilizar uma abordagem contextual.

---

# 32. Treinamento offline vs. operação online

É importante separar esses conceitos.

## Offline

Utiliza dados históricos para simular o comportamento do algoritmo.

```text
Dataset histórico
       ↓
Simulação
       ↓
Atualização dos parâmetros
       ↓
Avaliação
```

---

## Online

O algoritmo recebe novos clientes continuamente:

```text
Novo cliente
     ↓
Thompson Sampling
     ↓
Ação
     ↓
Interação real
     ↓
Reward
     ↓
Atualização
```

Essa é a situação para a qual algoritmos Bandit são especialmente apropriados.

---

# 33. API

Uma possível API pode receber os dados do cliente:

```json
{
    "age": 35,
    "job": "admin.",
    "marital": "single",
    "education": "university.degree",
    "housing": "yes",
    "loan": "no"
}
```

e retornar:

```json
{
    "recommended_action": "telefone"
}
```

Após a interação, outra chamada pode registrar o resultado:

```json
{
    "action": "telefone",
    "reward": 1
}
```

O algoritmo então atualiza:

```text
α
β
```

da ação correspondente.

---

# 34. Persistência do modelo

Como o Thompson Sampling possui um estado interno, é necessário persistir esse estado caso o serviço seja reiniciado.

Por exemplo:

```json
{
    "telefone": {
        "alpha": 101,
        "beta": 42
    },
    "email": {
        "alpha": 75,
        "beta": 68
    },
    "sms": {
        "alpha": 30,
        "beta": 40
    }
}
```

Esse estado pode ser armazenado em:

* arquivo JSON;
* banco de dados;
* Redis;
* outro mecanismo persistente.

Em uma arquitetura de produção, o armazenamento deve ser separado da aplicação para evitar perda do aprendizado.

---

# 35. Monitoramento

É importante monitorar o comportamento do algoritmo.

Algumas métricas recomendadas:

```text
reward acumulado
taxa de conversão
ações escolhidas
conversão por ação
quantidade de exploração
quantidade de explotação
α por ação
β por ação
```

Também é importante verificar se uma ação está sendo escolhida praticamente o tempo todo.

Por exemplo:

```text
Telefone → 98%
E-mail   → 1%
SMS      → 1%
```

Isso pode ser esperado caso telefone seja realmente superior, mas também pode indicar problemas no aprendizado ou nos dados.

---

# 36. O que significa "treinar" um Thompson Sampling?

No contexto deste projeto, é mais correto pensar em:

```text
inicialização
      ↓
interações
      ↓
observação dos rewards
      ↓
atualização das distribuições
      ↓
novo conhecimento
```

em vez de:

```text
X_train
   ↓
fit()
   ↓
modelo treinado
```

O modelo está constantemente aprendendo.

Por isso, o estado:

```text
α
β
```

é uma parte essencial do "modelo".

---

# 37. Resumo da lógica

O Thompson Sampling pode ser resumido em cinco etapas:

```text
1. Inicializar as distribuições
           ↓
2. Amostrar uma probabilidade para cada ação
           ↓
3. Escolher a ação com maior amostra
           ↓
4. Observar a recompensa
           ↓
5. Atualizar a distribuição da ação escolhida
           ↓
        Repetir
```

Para recompensa binária:

```text
Sucesso → α += 1
Fracasso → β += 1
```

A cada nova interação, o algoritmo atualiza sua crença sobre cada ação.

---

# 38. Conceito fundamental

A principal ideia que deve ser preservada na implementação é:

> **Thompson Sampling não tenta simplesmente descobrir qual ação é melhor. Ele tenta tomar a melhor decisão possível enquanto ainda aprende qual ação é melhor.**

Essa característica diferencia um Multi-Armed Bandit de um modelo tradicional de classificação.

No contexto de marketing, isso permite transformar o problema de:

```text
"Qual é a probabilidade de um cliente converter?"
```

em:

```text
"Dado o que já aprendemos e a incerteza existente,
qual ação devemos apresentar a este cliente agora?"
```

Essa é a lógica fundamental por trás do uso de Thompson Sampling em sistemas de recomendação, campanhas de marketing, ofertas e otimização de canais.
