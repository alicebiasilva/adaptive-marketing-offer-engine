# Plataforma adaptativa de ofertas 

## DO OBJETIVO DO PROJETO:

A solução deve escolher, para cada cliente elegível, qual oferta ou mensagem 
apresentar, buscando aumentar a conversão ao mesmo tempo que continua 
explorando alternativas.

O Datathon propõe um desafio prático no domínio financeiro: projetar uma 
plataforma de experimentação adaptativa para ofertas, mensagens ou próximos 
passos em canais digitais.

O objetivo não é reproduzir um sistema bancário real, mas sim mostrar 
maturidade técnica baseada nos conhecimentos do curso: formular o problema, 
construir baselines, versionar dados, servir componentes, avaliar qualidade, monitorar 
risco, documentar limitações e explicar decisões para públicos técnicos e de negócio, 
considerando o seguinte caso: 

Uma instituição financeira digital precisa decidir, em diferentes canais, qual 
oferta, mensagem ou próximo passo apresentar para cada cliente elegível. Regras 
fixas e testes A/B longos desperdiçam tráfego, demoram para reagir a mudanças de 
contexto e dificultam a personalização responsável. Esse é o ponto central de uma 
abordagem adaptativa (como multi-armed bandit): identificar comportamentos 
distintos, equilibrar exploração e explotação e aprender com respostas observadas 
sem congelar a decisão em regras estáticas. 

## DO USO DOS DADOS:
Use uma base Kaggle compatível com marketing, ofertas, propensão, 
campanhas, recomendação ou conversão como referência factual. Não use dados 
reais de clientes, identificadores, patrimônio, renda, gênero, raça ou regras comerciais 
privadas. Mantenha decisões sensíveis com humano no loop e documente base legal, 
finalidade, minimização e retenção


## DAS ETAPAS DE DESENVOLVIMENTO:

1. [OK] Definir problema, ações e recompensa
2. [OK] Escolher e documentar a base
3. [OK] Criar repositório e ambiente
4. [OK] Fazer EDA
5. Criar pipeline de preparação
6. Treinar modelo de propensão
7. Definir braços e simulador
8. Implementar baseline
9. Implementar Thompson Sampling
10. Avaliar com várias simulações
11. Registrar no MLflow
12. Criar Golden Set
13. Criar API
14. Criar testes
15. Adicionar Docker
16. Finalizar README e apresentação