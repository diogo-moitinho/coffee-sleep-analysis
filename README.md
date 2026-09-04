# Coffee & Sleep

Análise dos fatores associados à qualidade do sono, usando o [Global Coffee Health Dataset](https://www.kaggle.com/datasets/uom190346a/global-coffee-health-dataset) (10.000 registros sintéticos, 20 países).

## Resultado

**O modelo não prevê a qualidade do sono — e isso é a conclusão, não uma falha.**

A análise identificou três problemas estruturais no dataset que invalidam qualquer modelo treinado sem tratá-los:

**Redundância.** `Sleep_Hours`, `Sleep_Quality` e `Stress_Level` carregam a mesma informação. O Cramér's V entre as duas categóricas é 1,000, com tabela de contingência perfeitamente diagonal — as duas foram criadas cortando `Sleep_Hours` em faixas. `Caffeine_mg` é `Coffee_Intake` multiplicada por uma constante (r = 0,9998).

**Vazamento pela ausência.** `Health_Issues` tem 59% de nulos, e a ausência não é aleatória: nenhum registro faltante tem `Sleep_Hours` abaixo de 6h. Saber que o campo está vazio determina que a classe é Good ou Excellent. Imputar `"None"` — a escolha intuitiva — converteria 59% dos dados num marcador do alvo.

**Ausência de sinal.** Removidas as colunas que vazam, a única variável com associação legítima ao alvo é `Coffee_Intake`, com epsilon² de 0,030 (~3% da variação). O modelo final atinge F1-macro de 0,273 contra baseline de 0,245.

Análises públicas do mesmo dataset reportam acurácia de 99,5% e AUC de 1,000 prevendo estresse alto — resultado que decorre inteiramente da construção dos dados.

## Método

Testes escolhidos conforme os pressupostos, não por padrão:

| Situação | Teste |
|---|---|
| 2 grupos, normal, variâncias iguais | T-Test de Student |
| 2 grupos, normal, variâncias diferentes | T-Test de Welch |
| 2 grupos, não normal | Mann-Whitney U |
| 3+ grupos, normal, variâncias iguais | ANOVA + Tukey HSD |
| 3+ grupos, demais casos | Kruskal-Wallis + Mann-Whitney pareado (Holm) |
| Categórica × categórica | Qui-Quadrado + resíduos padronizados |

Todos os p-valores recebem correção FDR entre variáveis, e os resultados são ordenados por tamanho de efeito. Com n = 10.000, significância estatística não implica relevância: `Heart_Rate` difere entre níveis de sono com p < 0,001 e epsilon² de 0,001 — um batimento por minuto.

## Estrutura

## Como rodar

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab projeto.ipynb
```

## Limitações

Os dados são sintéticos. As relações observadas refletem as regras de geração, não fenômenos reais — o trabalho vale como exercício metodológico, não como evidência sobre café e sono.
## Como rodar

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab projeto.ipynb
```

## Limitações

Os dados são sintéticos. As relações observadas refletem as regras de geração, não fenômenos reais — o trabalho vale como exercício metodológico, não como evidência sobre café e sono.
