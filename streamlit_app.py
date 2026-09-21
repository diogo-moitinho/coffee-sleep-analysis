"""Estudo de caso: café, sono e um dataset que não permite previsão.

Seis abas — resultado, redundância, vazamento, ausência de sinal, método e
checklist. Todos os números vêm das saídas salvas em notebooks/projeto.ipynb.
Nada é recalculado aqui: o dataset não é versionado.

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------------------------------- #
# Números medidos — todos vindos das saídas do notebook
# --------------------------------------------------------------------------- #

LINHAS, COLUNAS, PAISES = 10_000, 16, 20
NULOS_SAUDE = 5_941

ALVO = pd.DataFrame([
    {"Classe": "Poor", "n": 961},
    {"Classe": "Fair", "n": 2_050},
    {"Classe": "Good", "n": 5_637},
    {"Classe": "Excellent", "n": 1_352},
])

CORRELACOES = pd.DataFrame([
    {"Par": "Caffeine_mg × Coffee_Intake",  "Pearson":  0.9998},
    {"Par": "Stress_Level × Sleep_Quality", "Pearson": -0.9121},
    {"Par": "Sleep_Quality × Sleep_Hours",  "Pearson":  0.9110},
    {"Par": "Stress_Level × Sleep_Hours",   "Pearson": -0.7935},
    {"Par": "Health_Issues × Age",          "Pearson":  0.2972},
    {"Par": "Health_Issues × BMI",          "Pearson":  0.2358},
    {"Par": "Sleep_Hours × Coffee_Intake",  "Pearson": -0.1903},
    {"Par": "Sleep_Quality × Coffee_Intake", "Pearson": -0.1744},
])

CRAMER = pd.DataFrame([
    {"Par": "Sleep_Quality × Stress_Level",  "V": 1.0000, "Magnitude": "GRANDE"},
    {"Par": "Sleep_Quality × Health_Issues", "V": 0.1049, "Magnitude": "PEQUENA"},
    {"Par": "Stress_Level × Health_Issues",  "V": 0.1047, "Magnitude": "PEQUENA"},
    {"Par": "Country × Health_Issues",       "V": 0.0730, "Magnitude": "DESPREZÍVEL"},
    {"Par": "Country × Smoking",             "V": 0.0547, "Magnitude": "DESPREZÍVEL"},
    {"Par": "Country × Sleep_Quality",       "V": 0.0486, "Magnitude": "DESPREZÍVEL"},
    {"Par": "Gender × Country",              "V": 0.0392, "Magnitude": "DESPREZÍVEL"},
])

EFEITOS = pd.DataFrame([
    {"Associação": "Sleep_Hours por Sleep_Quality", "Efeito": 0.8081, "Magnitude": "GRANDE"},
    {"Associação": "Sleep_Hours por Stress_Level",  "Efeito": 0.6484, "Magnitude": "GRANDE"},
    {"Associação": "Age por Health_Issues",         "Efeito": 0.0805, "Magnitude": "MÉDIA"},
    {"Associação": "BMI por Health_Issues",         "Efeito": 0.0479, "Magnitude": "PEQUENA"},
    {"Associação": "Caffeine_mg por Sleep_Quality", "Efeito": 0.0302, "Magnitude": "PEQUENA"},
    {"Associação": "Coffee_Intake por Sleep_Quality", "Efeito": 0.0302, "Magnitude": "PEQUENA"},
    {"Associação": "Heart_Rate por Sleep_Quality",  "Efeito": 0.0010, "Magnitude": "DESPREZÍVEL"},
])

# Distribuição de Sleep_Quality conforme Health_Issues estar preenchido ou não
AUSENCIA = pd.DataFrame([
    {"Registro": "Campo preenchido", "Classe": "Poor",      "Proporção": 23.7},
    {"Registro": "Campo preenchido", "Classe": "Fair",      "Proporção": 50.5},
    {"Registro": "Campo preenchido", "Classe": "Good",      "Proporção": 20.3},
    {"Registro": "Campo preenchido", "Classe": "Excellent", "Proporção":  5.5},
    {"Registro": "Campo vazio",      "Classe": "Poor",      "Proporção":  0.0},
    {"Registro": "Campo vazio",      "Classe": "Fair",      "Proporção":  0.0},
    {"Registro": "Campo vazio",      "Classe": "Good",      "Proporção": 81.0},
    {"Registro": "Campo vazio",      "Classe": "Excellent", "Proporção": 19.0},
])

HORAS_SONO = pd.DataFrame([
    {"Registro": "Campo preenchido", "Medida": "Mínimo", "Horas": 3.0},
    {"Registro": "Campo preenchido", "Medida": "Média",  "Horas": 5.75},
    {"Registro": "Campo preenchido", "Medida": "Máximo", "Horas": 10.0},
    {"Registro": "Campo vazio",      "Medida": "Mínimo", "Horas": 6.0},
    {"Registro": "Campo vazio",      "Medida": "Média",  "Horas": 7.24},
    {"Registro": "Campo vazio",      "Medida": "Máximo", "Horas": 10.0},
])

BASELINE, MODELO, ARVORES = 0.245, 0.272, 341
F1_TREINO, F1_TESTE = 0.928, 0.272

REGULARIZACAO = pd.DataFrame([
    {"Configuração": "31 folhas<br>30 amostras",  "Treino": 0.831, "Teste": 0.270},
    {"Configuração": "15 folhas<br>100 amostras", "Treino": 0.596, "Teste": 0.267},
    {"Configuração": "7 folhas<br>300 amostras",  "Treino": 0.407, "Teste": 0.256},
])

POR_CLASSE = pd.DataFrame([
    {"Classe": "Poor",      "F1": 0.10, "Suporte":  240},
    {"Classe": "Fair",      "F1": 0.22, "Suporte":  513},
    {"Classe": "Good",      "F1": 0.56, "Suporte": 1_409},
    {"Classe": "Excellent", "F1": 0.21, "Suporte":  338},
])

TESTES = pd.DataFrame([
    {"Situação": "2 grupos, normal, variâncias iguais",      "Teste": "T-Test de Student"},
    {"Situação": "2 grupos, normal, variâncias diferentes",  "Teste": "T-Test de Welch"},
    {"Situação": "2 grupos, não normal",                     "Teste": "Mann-Whitney U"},
    {"Situação": "3+ grupos, normal, variâncias iguais",     "Teste": "ANOVA + Tukey HSD"},
    {"Situação": "3+ grupos, demais casos",                  "Teste": "Kruskal-Wallis + Mann-Whitney (Holm)"},
    {"Situação": "Categórica × categórica",                  "Teste": "Qui-Quadrado + resíduos padronizados"},
])

VERDE, VERMELHO, LARANJA, AZUL, CINZA = (
    "#0ca30c", "#d03b3b", "#ec835a", "#2a78d6", "#9aa0a6")

COR_MAGNITUDE = {"GRANDE": VERMELHO, "MÉDIA": LARANJA,
                 "PEQUENA": AZUL, "DESPREZÍVEL": CINZA}

# --------------------------------------------------------------------------- #
# Auxiliares
# --------------------------------------------------------------------------- #

def num(v: float, casas: int = 0) -> str:
    return f"{v:,.{casas}f}".replace(",", "§").replace(".", ",").replace("§", ".")


def pct(v: float, casas: int = 1) -> str:
    return f"{v:.{casas}f}".replace(".", ",") + "%"


def mostrar(fig, altura: int = 300) -> None:
    """Enxoval comum de todos os gráficos."""
    fig.update_layout(height=altura, margin=dict(l=0, r=10, t=10, b=0),
                      separators=",.")
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------- #
# Página
# --------------------------------------------------------------------------- #

st.set_page_config(page_title="Café e sono — análise", layout="wide")

st.title("Café, sono e um dataset que não permite previsão")
st.caption(
    "Análise do Global Coffee Health Dataset — 10.000 registros sintéticos, 20 "
    "países. O objetivo era prever a qualidade do sono. O resultado foi descobrir "
    "por que isso não é possível com esses dados."
)

abas = st.tabs(["O resultado", "Redundância", "Vazamento pela ausência",
                "Ausência de sinal", "Método", "O que fica"])

# ═══ 1. O RESULTADO ════════════════════════════════════════════════════════ #

with abas[0]:
    st.error("**O modelo não prevê a qualidade do sono** — e o valor do projeto "
             "está em mostrar exatamente por quê.")

    c = st.columns(4)
    c[0].metric("Registros", num(LINHAS))
    c[1].metric("Colunas", COLUNAS)
    c[2].metric("F1-macro do modelo", num(MODELO, 3))
    c[3].metric("F1-macro do baseline", num(BASELINE, 3),
                f"+{num(MODELO - BASELINE, 3)} do modelo")

    st.markdown(
        "Um chute estratificado acerta 0,245. O LightGBM, com "
        f"{ARVORES} árvores, chega a {num(MODELO, 3)}. **O ganho de todo o esforço "
        "de modelagem é de 0,027** — e a análise mostra que isso não é falha de "
        "ajuste, é ausência de padrão para aprender."
    )

    st.divider()
    st.subheader("Três problemas estruturais no dataset")

    col = st.columns(3)
    problemas = [
        ("Redundância",
         "Sleep_Hours, Sleep_Quality e Stress_Level carregam a mesma informação. "
         "O Cramér's V entre as duas categóricas é **1,000**: foram criadas "
         "cortando a mesma coluna em faixas.", "Redundância"),
        ("Vazamento pela ausência",
         f"Health_Issues tem {pct(NULOS_SAUDE / LINHAS * 100)} de nulos. **Nenhum "
         "registro faltante tem menos de 6 horas de sono.** Saber que o campo está "
         "vazio já determina a classe.", "Vazamento"),
        ("Ausência de sinal",
         "Removidas as colunas que vazam, a única associação legítima com o alvo é "
         "Coffee_Intake, com epsilon² de **0,030** — cerca de 3% da variação.",
         "Sinal"),
    ]
    for coluna, (titulo, texto, _) in zip(col, problemas):
        with coluna.container(border=True):
            st.markdown(f"**{titulo}**")
            st.markdown(texto)

    st.divider()
    st.subheader("A distribuição do alvo")

    fig = px.bar(ALVO, x="Classe", y="n", text=[num(v) for v in ALVO["n"]])
    fig.update_traces(marker_color=AZUL, textposition="outside", cliponaxis=False)
    fig.update_layout(yaxis_title="registros", xaxis_title=None)
    mostrar(fig, 280)

    st.caption(
        "A classe Good sozinha é 56% da base. Por isso a métrica é F1-macro, que "
        "pesa as quatro classes igualmente — a acurácia simples ficaria em 40% só "
        "chutando a classe majoritária com alguma frequência."
    )

# ═══ 2. REDUNDÂNCIA ════════════════════════════════════════════════════════ #

with abas[1]:
    st.subheader("Colunas que são a mesma coluna")

    corr = CORRELACOES.copy()
    corr["abs"] = corr["Pearson"].abs()
    corr["Tipo"] = ["Redundante" if v >= 0.7 else "Associação real"
                    for v in corr["abs"]]

    fig = px.bar(corr.sort_values("abs"), x="abs", y="Par", orientation="h",
                 text=[num(v, 4) for v in corr.sort_values("abs")["Pearson"]],
                 color="Tipo", range_x=[0, 1.18],
                 color_discrete_map={"Redundante": VERMELHO,
                                     "Associação real": AZUL})
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(xaxis_title="correlação de Pearson (módulo)",
                      yaxis_title=None, legend_title=None)
    mostrar(fig, 380)

    st.markdown(
        "`Caffeine_mg` é `Coffee_Intake` multiplicada por uma constante — "
        "**r = 0,9998**. Não é uma variável a mais, é a mesma variável em outra "
        "unidade. E `Sleep_Quality`, `Sleep_Hours` e `Stress_Level` formam um bloco "
        "onde qualquer uma prevê as outras quase perfeitamente."
    )

    st.divider()
    st.subheader("Associação entre as categóricas")

    fig = px.bar(CRAMER.sort_values("V"), x="V", y="Par", orientation="h",
                 text=[num(v, 4) for v in CRAMER.sort_values("V")["V"]],
                 color="Magnitude", range_x=[0, 1.2],
                 color_discrete_map=COR_MAGNITUDE)
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(xaxis_title="Cramér's V", yaxis_title=None,
                      legend_title=None)
    mostrar(fig, 340)

    st.warning(
        "**Cramér's V de 1,000 é o alerta máximo.** Significa que a tabela de "
        "contingência é perfeitamente diagonal: cada nível de Stress_Level "
        "corresponde a exatamente um nível de Sleep_Quality. Treinar um modelo com "
        "as duas é entregar a resposta na entrada."
    )

    st.caption(
        "Repare no contraste: tirando esse par e os dois de Health_Issues, todo o "
        "resto fica abaixo de 0,08 — desprezível. O dataset tem pouquíssima "
        "estrutura além da que foi construída artificialmente."
    )

# ═══ 3. VAZAMENTO PELA AUSÊNCIA ════════════════════════════════════════════ #

with abas[2]:
    st.subheader("O campo vazio entrega a resposta")

    c = st.columns(3)
    c[0].metric("Nulos em Health_Issues", num(NULOS_SAUDE))
    c[1].metric("Proporção da base", pct(NULOS_SAUDE / LINHAS * 100))
    c[2].metric("Faltantes com sono Poor ou Fair", "0")

    fig = px.bar(AUSENCIA, x="Proporção", y="Registro", color="Classe",
                 orientation="h", text=[pct(v) if v else "" for v in AUSENCIA["Proporção"]],
                 color_discrete_map={"Poor": VERMELHO, "Fair": LARANJA,
                                     "Good": AZUL, "Excellent": VERDE})
    fig.update_traces(textposition="inside")
    fig.update_layout(barmode="stack", xaxis_title="% dos registros do grupo",
                      yaxis_title=None, legend_title=None, bargap=0.45)
    mostrar(fig, 250)

    st.error(
        "Quando `Health_Issues` está vazio, **100% dos registros são Good ou "
        "Excellent**. Nenhum Poor, nenhum Fair. A ausência do dado não é ruído: é "
        "um preditor perfeito de metade do alvo."
    )

    st.divider()
    st.subheader("Horas de sono nos dois grupos")

    fig = px.bar(HORAS_SONO, x="Medida", y="Horas", color="Registro",
                 barmode="group", text=[num(v, 2) for v in HORAS_SONO["Horas"]],
                 color_discrete_map={"Campo preenchido": AZUL,
                                     "Campo vazio": VERMELHO})
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(xaxis_title=None, yaxis_title="horas de sono",
                      legend_title=None)
    mostrar(fig, 300)

    st.markdown(
        "O mínimo conta a história: quem tem o campo preenchido pode dormir "
        "**3 horas**; quem tem o campo vazio nunca dorme menos de **6**. O gerador "
        "do dataset só omitiu o histórico de saúde de quem dorme bem."
    )

    st.caption(
        "Esse tipo de vazamento é traiçoeiro porque não aparece em nenhuma matriz "
        "de correlação: a informação não está no valor da coluna, está no fato de "
        "ela estar vazia. Um `fillna` silencioso teria apagado a evidência e "
        "mantido o efeito."
    )

# ═══ 4. AUSÊNCIA DE SINAL ══════════════════════════════════════════════════ #

with abas[3]:
    st.subheader("O que sobra depois de remover o que vaza")

    fig = px.bar(EFEITOS.sort_values("Efeito"), x="Efeito", y="Associação",
                 orientation="h", color="Magnitude", range_x=[0, 1],
                 text=[num(v, 4) for v in EFEITOS.sort_values("Efeito")["Efeito"]],
                 color_discrete_map=COR_MAGNITUDE)
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(xaxis_title="epsilon² (proporção da variação explicada)",
                      yaxis_title=None, legend_title=None)
    mostrar(fig, 340)

    st.markdown(
        "As duas barras grandes são justamente as colunas redundantes. Removidas "
        "elas, o maior efeito legítimo é **Coffee_Intake com 0,030** — 3% da "
        "variação. O resto é ruído estatisticamente significativo e praticamente "
        "irrelevante."
    )

    st.info(
        "Com n = 10.000, significância não implica relevância. Heart_Rate difere "
        "entre níveis de sono com p < 0,001 e epsilon² de 0,001 — a diferença real "
        "é de **um batimento por minuto**."
    )

    st.divider()
    st.subheader("Modelo contra baseline")

    comp = pd.DataFrame({
        "Abordagem": ["Chute estratificado", f"LightGBM ({ARVORES} árvores)"],
        "F1-macro": [BASELINE, MODELO],
    })
    fig = px.bar(comp, x="F1-macro", y="Abordagem", orientation="h",
                 text=[num(BASELINE, 3), num(MODELO, 3)], range_x=[0, 0.34],
                 color="Abordagem", color_discrete_map={
                     "Chute estratificado": CINZA,
                     f"LightGBM ({ARVORES} árvores)": AZUL})
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False, yaxis_title=None)
    mostrar(fig, 200)

    st.divider()
    st.subheader("A prova: regularizar não recupera desempenho")

    reg = REGULARIZACAO.melt(id_vars="Configuração",
                             var_name="Conjunto", value_name="F1-macro")
    fig = px.line(reg, x="Configuração", y="F1-macro", color="Conjunto",
                  markers=True, range_y=[0, 0.95],
                  color_discrete_map={"Treino": LARANJA, "Teste": AZUL})
    fig.update_traces(line=dict(width=2), marker=dict(size=9))
    fig.update_layout(xaxis_title=None, legend_title=None)
    mostrar(fig, 330)

    st.markdown(
        f"O modelo sem freio decora: F1 de **{num(F1_TREINO, 3)}** no treino contra "
        f"**{num(F1_TESTE, 3)}** no teste. Apertando a regularização — de 31 para 7 "
        "folhas, de 30 para 300 amostras mínimas — o treino despenca de 0,831 para "
        "0,407 e **o teste não se move**: 0,270 → 0,256."
    )
    st.success(
        "Reduzir o overfitting em dois terços não recuperou desempenho algum. É "
        "esse experimento que separa *“o modelo está mal ajustado”* de *“não há o "
        "que aprender”*. A curva de aprendizado confirma: a linha de validação é "
        "reta entre 800 e 4.000 amostras de treino."
    )

    st.divider()
    st.subheader("Desempenho por classe")

    fig = px.bar(POR_CLASSE, x="Classe", y="F1", text=[num(v, 2) for v in POR_CLASSE["F1"]],
                 color="Classe", color_discrete_map={
                     "Poor": VERMELHO, "Fair": LARANJA,
                     "Good": AZUL, "Excellent": VERDE})
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False, yaxis_title="F1", xaxis_title=None,
                      yaxis_range=[0, 0.68])
    mostrar(fig, 280)

    st.caption(
        "O modelo só acerta a classe majoritária. Nas outras três fica em torno de "
        "0,10 a 0,22 — praticamente o acaso. A acurácia global de 40% esconde isso; "
        "o F1-macro não."
    )

# ═══ 5. MÉTODO ═════════════════════════════════════════════════════════════ #

with abas[4]:
    st.subheader("Testes escolhidos pelos pressupostos, não por padrão")

    st.dataframe(TESTES, hide_index=True, use_container_width=True)

    st.markdown(
        "A rotina verifica normalidade e homogeneidade de variância antes de "
        "decidir qual teste aplicar, em vez de rodar um T-Test em tudo. Para "
        "categóricas, o Qui-Quadrado vem acompanhado dos resíduos padronizados, "
        "que mostram **qual célula** puxa a associação."
    )

    st.divider()
    st.subheader("Duas decisões que mudam a leitura")

    c = st.columns(2)
    with c[0].container(border=True):
        st.markdown("**Correção FDR entre variáveis**")
        st.markdown(
            "Testar dezenas de pares infla o número de falsos positivos. Todos os "
            "p-valores passam por correção de taxa de descoberta falsa antes de "
            "qualquer conclusão."
        )
    with c[1].container(border=True):
        st.markdown("**Ordenação por tamanho de efeito**")
        st.markdown(
            "Os resultados são ordenados por epsilon² ou Cramér's V, não por "
            "p-valor. Com n grande, o p-valor mede sobretudo o tamanho da amostra; "
            "o tamanho de efeito mede o que importa."
        )

    st.divider()
    st.subheader("Partição e alvo")

    c = st.columns(3)
    c[0].metric("Treino", num(6_000))
    c[1].metric("Validação", num(1_500))
    c[2].metric("Teste", num(2_500))

    st.caption(
        "Onze features entraram no modelo, incluindo uma Faixa_Etaria derivada. "
        "As colunas identificadas como redundantes ou vazadas ficaram de fora."
    )

# ═══ 6. O QUE FICA ═════════════════════════════════════════════════════════ #

with abas[5]:
    st.subheader("Checklist derivado deste projeto")

    for i, item in enumerate([
        "Testar associação entre categóricas **antes** de modelar — Cramér's V "
        "próximo de 1 indica coluna duplicada.",
        "Verificar se a **ausência** de dados prediz o alvo, não só o valor.",
        "Comparar sempre com um baseline. Sem ele, 0,272 parece um número.",
        "Ordenar achados por tamanho de efeito, não por p-valor.",
        "Diante de overfitting, regularizar e observar se o teste sobe. **Se não "
        "subir, o problema é ausência de sinal**, não de ajuste.",
    ], 1):
        st.markdown(f"**{i}.** {item}")

    st.divider()
    st.subheader("Limitações")

    st.warning(
        "Os dados são **sintéticos**. As relações observadas refletem as regras de "
        "geração do dataset, não fenômenos reais sobre café e sono. O trabalho vale "
        "como exercício metodológico, não como evidência sobre saúde."
    )

    st.markdown(
        "Vale notar que os três problemas encontrados — redundância, vazamento pela "
        "ausência e falta de sinal — são **artefatos do gerador**. Em dados reais "
        "eles apareceriam por outros motivos, mas os testes que os revelam seriam "
        "os mesmos."
    )

st.divider()
st.caption(
    "Números extraídos das saídas de notebooks/projeto.ipynb · "
    "github.com/diogo-moitinho/coffee-sleep-analysis"
)
