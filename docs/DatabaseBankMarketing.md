# Dicionário de Dados — Bank Marketing Dataset

## Visão geral

A base utilizada contém informações de clientes de uma instituição bancária e o resultado de campanhas de marketing direto realizadas para oferta de produtos financeiros.
Ela é baseada em dados reais de uma campanha de marketing bancário, mas com algumas características importantes sobre privacidade e anonimização.

Ela foi criada a partir de campanhas de marketing direto de uma instituição bancária portuguesa, coletadas entre maio de 2008 e novembro de 2010. O objetivo era prever se um cliente aceitaria uma oferta de depósito a prazo (term deposit) após ser contatado por telefone.

Cada linha da base representa um cliente que recebeu uma abordagem da campanha.

A variável alvo é:

- **y**: indica se o cliente aceitou ou não a oferta realizada.

> Observação: a variável `duration` representa a duração do último contato e deve ser removida durante a modelagem devido ao risco de vazamento de informação (data leakage), pois essa informação só estaria disponível após o contato com o cliente.

---

# Dicionário das Variáveis

| Variável | Tipo | Descrição |
|---|---|---|
| age | Numérica | Idade do cliente em anos. |
| job | Categórica | Tipo de ocupação/profissão do cliente. Exemplos: admin, technician, entrepreneur, student. |
| marital | Categórica | Estado civil do cliente. Valores possíveis: married, single, divorced. |
| education | Categórica | Nível de escolaridade do cliente. |
| default | Categórica | Indica se o cliente possui histórico de inadimplência/crédito em atraso. |
| housing | Categórica | Indica se o cliente possui financiamento imobiliário. |
| loan | Categórica | Indica se o cliente possui empréstimo pessoal. |
| contact | Categórica | Tipo de canal utilizado para contato com o cliente. |
| month | Categórica | Mês em que ocorreu o último contato da campanha. |
| day_of_week | Categórica | Dia da semana do último contato. |
| duration | Numérica | Duração do último contato em segundos. Deve ser removida da modelagem por representar informação posterior ao início da interação. |
| campaign | Numérica | Quantidade de contatos realizados durante a campanha atual para esse cliente. |
| pdays | Numérica | Número de dias desde o último contato de uma campanha anterior. |
| previous | Numérica | Quantidade de contatos realizados antes da campanha atual. |
| poutcome | Categórica | Resultado da campanha anterior. |
| emp.var.rate | Numérica | Taxa de variação do emprego (indicador econômico). |
| cons.price.idx | Numérica | Índice de preços ao consumidor (indicador econômico). |
| cons.conf.idx | Numérica | Índice de confiança do consumidor (indicador econômico). |
| euribor3m | Numérica | Taxa Euribor de 3 meses, indicador econômico europeu. |
| nr.employed | Numérica | Número de empregados no setor econômico no período da campanha. |
| y | Binária (target) | Resultado da campanha: cliente aceitou (`yes`) ou recusou (`no`) a oferta. |

---

# Classificação das variáveis

## Variáveis demográficas

Utilizadas para identificar características gerais do cliente:

- age
- job
- marital
- education

---

## Variáveis de relacionamento financeiro

Representam produtos e histórico financeiro:

- default
- housing
- loan

---

## Variáveis da campanha

Relacionadas à estratégia de contato:

- contact
- month
- day_of_week
- campaign
- pdays
- previous
- poutcome

---

## Indicadores econômicos

Representam o contexto macroeconômico no momento da campanha:

- emp.var.rate
- cons.price.idx
- cons.conf.idx
- euribor3m
- nr.employed

---

## Variável alvo

### y — Conversão da campanha

Valores:

| Valor | Significado |
|---|---|
| yes | Cliente aceitou a oferta |
| no | Cliente não aceitou a oferta |

Essa variável será utilizada para calcular a recompensa dos algoritmos adaptativos:

- Recompensa = 1 → conversão ocorreu
- Recompensa = 0 → conversão não ocorreu

## Limitação temporal

A base Bank Marketing possui a variável `month`, indicando o mês da campanha, porém não disponibiliza o ano ou uma data completa para cada observação.

A fonte original informa que os dados foram coletados entre 2008 e 2010, mas não é possível reconstruir a linha temporal exata de cada registro.

Por esse motivo, o projeto utiliza as variáveis de contexto econômico disponíveis (`emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m` e `nr.employed`) para representar o cenário de mercado no momento da decisão.