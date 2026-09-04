"""Comparação de variáveis entre grupos, com a conclusão em português.

A saída padrão é curta: diz o que foi encontrado, o quanto isso é forte e
se é confiável. O detalhe técnico (teste escolhido, pressupostos, tabela
completa de pares) só aparece com detalhes=True.

    comparar_dois_grupos(df, variavel, grupo)   -> 2 grupos, variável numérica
    comparar_n_grupos(df, variavel, grupo)      -> 3+ grupos, variável numérica
    comparar_categoricas(df, variavel, grupo)   -> duas variáveis categóricas
"""

from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import (
    chi2_contingency, f_oneway, kruskal, levene, mannwhitneyu,
    normaltest, shapiro, ttest_ind,
)
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

LIMIAR_RESIDUO = 1.96
LIMITE_SHAPIRO = 5000     # acima disso Shapiro fica sensível demais
MIN_OBS_NORMALIDADE = 20  # abaixo disso não dá para avaliar normalidade

# Faixas de referência por métrica: (pequeno, médio, grande).
FAIXAS = {
    "Cohen's d": (0.2, 0.5, 0.8),
    "Rank-biserial": (0.1, 0.3, 0.5),
    "Eta²": (0.01, 0.06, 0.14),
    "Epsilon²": (0.01, 0.06, 0.14),
    "Cramér's V": (0.1, 0.3, 0.5),
}
# Acima disso a associação é forte demais para ser real: quase sempre
# significa que uma variável foi derivada da outra.
LIMIAR_REDUNDANCIA = 0.90


def _magnitude(efeito, metrica):
    """Traduz o tamanho de efeito em uma palavra."""
    if efeito is None or np.isnan(efeito):
        return "indefinida"
    pequeno, medio, grande = FAIXAS[metrica]
    valor = abs(efeito)
    if valor < pequeno:
        return "DESPREZÍVEL"
    if valor < medio:
        return "PEQUENA"
    if valor < grande:
        return "MÉDIA"
    return "GRANDE"


def _formatar_p(p_valor):
    """p-valor legível, sem notação científica assustadora."""
    if p_valor < 0.001:
        return "p < 0.001"
    return f"p = {p_valor:.3f}"


def _cabecalho(titulo):
    print(f"\n{'─' * 64}\n{titulo}\n{'─' * 64}")


def _separar(df, variavel, grupo, ordem=None):
    """Devolve (nomes dos grupos, lista de Series com os valores)."""
    nomes = ordem if ordem is not None else sorted(df[grupo].dropna().unique())
    amostras = [df.loc[df[grupo] == nome, variavel].dropna() for nome in nomes]
    return list(nomes), amostras


def _diagnosticar(amostras, alpha, detalhes=False):
    """Testa normalidade (por grupo) e homocedasticidade (no conjunto).

    A normalidade é avaliada dentro de cada grupo, não na variável inteira:
    a distribuição global pode parecer bimodal só porque os grupos têm
    médias diferentes.
    """
    if any(len(a) < MIN_OBS_NORMALIDADE for a in amostras):
        normal, p_valores, nome = False, [], "amostra pequena"
    else:
        p_valores = []
        for a in amostras:
            teste = shapiro if len(a) <= LIMITE_SHAPIRO else normaltest
            p_valores.append(teste(a)[1])
        normal = all(p >= alpha for p in p_valores)
        nome = "Shapiro-Wilk" if len(amostras[0]) <= LIMITE_SHAPIRO else "D'Agostino"

    _, p_levene = levene(*amostras, center="median")
    variancias_iguais = p_levene > alpha

    if detalhes:
        print(f"\n  Pressupostos:")
        if p_valores:
            print(f"    Normalidade ({nome}): {'NORMAL' if normal else 'NÃO NORMAL'} "
                  f"| p por grupo = {[f'{p:.4f}' for p in p_valores]}")
        else:
            print(f"    Normalidade: INDETERMINADA ({nome}) -> assumindo não normal")
        print(f"    Variâncias (Levene): "
              f"{'HOMOGÊNEAS' if variancias_iguais else 'HETEROGÊNEAS'} | p = {p_levene:.4f}")

    return normal, variancias_iguais


def _alerta_redundancia(efeito, metrica, variavel, grupo):
    """Avisa quando a associação é forte demais para ser um achado real."""
    if abs(efeito) >= LIMIAR_REDUNDANCIA and metrica in ("Cramér's V", "Eta²", "Epsilon²"):
        print(f"\n  ATENÇÃO: associação quase perfeita. Verifique se '{variavel}' "
              f"não foi\n  derivada de '{grupo}' (ou vice-versa) — nesse caso o teste "
              f"apenas\n  redescobre a regra que criou a coluna.")


# --------------------------------------------------------------------------- #
# 2 grupos
# --------------------------------------------------------------------------- #

def comparar_dois_grupos(df, variavel, grupo, alpha=0.05, ordem=None, detalhes=False):
    """Compara uma variável numérica entre exatamente 2 grupos.

    Normal + variâncias iguais      -> T-Test de Student
    Normal + variâncias diferentes  -> T-Test de Welch
    Não normal                      -> Mann-Whitney U
    """
    if not pd.api.types.is_numeric_dtype(df[variavel]):
        raise TypeError(f"'{variavel}' não é numérica. Use comparar_categoricas().")

    nomes, amostras = _separar(df, variavel, grupo, ordem)
    if len(nomes) != 2:
        raise ValueError(f"Esperados 2 grupos, encontrados {len(nomes)}: {nomes}")

    a, b = amostras
    _cabecalho(f"{variavel} — {nomes[0]} (n={len(a)}) vs {nomes[1]} (n={len(b)})")

    normal, variancias_iguais = _diagnosticar(amostras, alpha, detalhes)

    if normal:
        teste = "T-Test de Student" if variancias_iguais else "T-Test de Welch"
        resultado = ttest_ind(b, a, equal_var=variancias_iguais)
        p_valor, ic = resultado.pvalue, resultado.confidence_interval(1 - alpha)
        diferenca = b.mean() - a.mean()
        var_pool = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
        efeito, metrica, base = diferenca / np.sqrt(var_pool), "Cohen's d", "média"
    else:
        teste = "Mann-Whitney U"
        estatistica, p_valor = mannwhitneyu(b, a, alternative="two-sided")
        ic = None
        diferenca = b.median() - a.median()
        efeito = 2 * estatistica / (len(a) * len(b)) - 1
        metrica, base = "Rank-biserial", "mediana"

    significativo = p_valor < alpha
    magnitude = _magnitude(efeito, metrica)

    # --- conclusão em português ---
    maior, menor = (nomes[1], nomes[0]) if diferenca > 0 else (nomes[0], nomes[1])
    print(f"\n  {nomes[0]}: {base} {a.mean() if normal else a.median():.2f}"
          f"   |   {nomes[1]}: {base} {b.mean() if normal else b.median():.2f}")

    if significativo:
        print(f"\n  {maior} tem {variavel} MAIOR que {menor}, "
              f"por {abs(diferenca):.2f}.")
        print(f"  Essa diferença é {magnitude} e é confiável ({_formatar_p(p_valor)}).")
        if magnitude == "DESPREZÍVEL":
            print(f"  Mesmo confiável, a diferença é pequena demais para ter uso prático.")
    else:
        print(f"\n  Não há diferença entre {nomes[0]} e {nomes[1]} ({_formatar_p(p_valor)}).")

    if detalhes:
        print(f"\n  Detalhe técnico:")
        print(f"    Teste: {teste} | {_formatar_p(p_valor)} (alpha = {alpha})")
        print(f"    Diferença de {base}s ({nomes[1]} - {nomes[0]}): {diferenca:+.4f}")
        if ic is not None:
            print(f"    IC {int((1 - alpha) * 100)}%: [{ic.low:+.4f}, {ic.high:+.4f}]")
        print(f"    {metrica} = {efeito:+.3f}")

    return {
        "variavel": variavel, "grupos": nomes, "teste": teste,
        "normal": normal, "variancias_iguais": variancias_iguais,
        "p_valor": p_valor, "diferenca": diferenca,
        "ic": (ic.low, ic.high) if ic is not None else None,
        "metrica_efeito": metrica, "tamanho_efeito": efeito,
        "magnitude": magnitude, "significativo": significativo,
    }


# --------------------------------------------------------------------------- #
# N grupos
# --------------------------------------------------------------------------- #

def _posthoc_tukey(nomes, amostras, alpha):
    """Tukey HSD: já devolve diferenças, ICs e p ajustado."""
    valores = np.concatenate([a.to_numpy() for a in amostras])
    rotulos = np.concatenate([[str(n)] * len(a) for n, a in zip(nomes, amostras)])
    bruto = pairwise_tukeyhsd(valores, rotulos, alpha=alpha)
    return pd.DataFrame(
        [(l[0], l[1], l[2], l[4], l[5], l[3], bool(l[6])) for l in bruto.summary().data[1:]],
        columns=["grupo_1", "grupo_2", "diferenca", "ic_inferior", "ic_superior",
                 "p_ajustado", "significativo"],
    )


def _posthoc_mannwhitney(nomes, amostras, alpha, metodo="holm"):
    """Mann-Whitney par a par com correção do erro familiar.

    Alternativa ao teste de Dunn sem exigir scikit-posthocs. Sem IC: no caso
    não-paramétrico ele exigiria bootstrap.
    """
    linhas, p_valores = [], []
    for i, j in combinations(range(len(nomes)), 2):
        _, p = mannwhitneyu(amostras[j], amostras[i], alternative="two-sided")
        linhas.append([nomes[i], nomes[j], amostras[j].median() - amostras[i].median()])
        p_valores.append(p)
    rejeita, p_ajustado, _, _ = multipletests(p_valores, alpha=alpha, method=metodo)
    tabela = pd.DataFrame(linhas, columns=["grupo_1", "grupo_2", "diferenca"])
    tabela["ic_inferior"] = np.nan
    tabela["ic_superior"] = np.nan
    tabela["p_ajustado"] = p_ajustado
    tabela["significativo"] = rejeita
    return tabela


def comparar_n_grupos(df, variavel, grupo, alpha=0.05, ordem=None, detalhes=False):
    """Compara uma variável numérica entre 3 ou mais grupos.

    Normal + variâncias iguais  -> ANOVA de uma via
    Caso contrário              -> Kruskal-Wallis

    Retorna (resumo, tabela_posthoc).
    """
    if not pd.api.types.is_numeric_dtype(df[variavel]):
        raise TypeError(f"'{variavel}' não é numérica. Use comparar_categoricas().")

    nomes, amostras = _separar(df, variavel, grupo, ordem)
    if len(nomes) < 3:
        raise ValueError(f"Use comparar_dois_grupos para {len(nomes)} grupos.")

    k, n_total = len(nomes), sum(len(a) for a in amostras)
    _cabecalho(f"{variavel} por {grupo} — {k} grupos, n={n_total}")

    normal, variancias_iguais = _diagnosticar(amostras, alpha, detalhes)

    if normal and variancias_iguais:
        teste = "ANOVA de uma via"
        estatistica, p_valor = f_oneway(*amostras)
        # Eta²: proporção da variância explicada pelo grupo.
        efeito, metrica = (estatistica * (k - 1)) / (estatistica * (k - 1) + n_total - k), "Eta²"
        centro = "média"
    else:
        # Com variâncias heterogêneas a ANOVA clássica infla o erro tipo I.
        teste = "Kruskal-Wallis"
        estatistica, p_valor = kruskal(*amostras)
        efeito, metrica = max(0.0, (estatistica - k + 1) / (n_total - k)), "Epsilon²"
        centro = "mediana"

    significativo = p_valor < alpha
    magnitude = _magnitude(efeito, metrica)

    # --- ranking dos grupos: a informação mais útil, em uma linha ---
    resumos = [(nome, a.mean() if centro == "média" else a.median(), len(a))
               for nome, a in zip(nomes, amostras)]
    ranking = sorted(resumos, key=lambda x: x[1])
    print(f"\n  {variavel} por grupo ({centro}, do menor para o maior):")
    print("    " + "  <  ".join(f"{nome} {valor:.2f}" for nome, valor, _ in ranking))

    resumo = {"variavel": variavel, "grupos": nomes, "teste": teste,
              "normal": normal, "variancias_iguais": variancias_iguais,
              "estatistica": estatistica, "p_valor": p_valor,
              "metrica_efeito": metrica, "tamanho_efeito": efeito,
              "magnitude": magnitude, "significativo": significativo}

    if not significativo:
        print(f"\n  Os grupos NÃO diferem em {variavel} ({_formatar_p(p_valor)}).")
        print(f"  A variação entre eles é compatível com o acaso.")
        if detalhes:
            print(f"\n  Detalhe técnico: {teste} | estatística = {estatistica:.4f}")
        return resumo, pd.DataFrame()

    if normal and variancias_iguais:
        tabela = _posthoc_tukey(nomes, amostras, alpha)
        nome_posthoc = "Tukey HSD"
    else:
        tabela = _posthoc_mannwhitney(nomes, amostras, alpha)
        nome_posthoc = "Mann-Whitney pareado (correção de Holm)"

    tabela = tabela.reindex(
        tabela["diferenca"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)
    diferentes = tabela[tabela["significativo"]]
    iguais = tabela[~tabela["significativo"]]

    # --- conclusão em português ---
    menor_nome, menor_valor, _ = ranking[0]
    maior_nome, maior_valor, _ = ranking[-1]
    print(f"\n  Os grupos DIFEREM em {variavel}. "
          f"A diferença é {magnitude} ({metrica} = {efeito:.3f}, {_formatar_p(p_valor)}).")
    print(f"  Do menor para o maior: {menor_nome} ({menor_valor:.2f}) "
          f"até {maior_nome} ({maior_valor:.2f}), "
          f"uma distância de {maior_valor - menor_valor:.2f}.")

    if len(diferentes) == len(tabela):
        print(f"  Todos os {len(tabela)} pares de grupos diferem entre si.")
    elif len(diferentes) == 0:
        # O teste global passou raspando, mas nenhum par sobrevive à correção.
        print(f"  Porém NENHUM par específico sobrevive à correção: o resultado "
              f"global é frágil\n  e não dá para apontar quais grupos diferem.")
    else:
        print(f"\n  {len(diferentes)} dos {len(tabela)} pares diferem. "
              f"Os pares SEM diferença são:")
        for _, linha in iguais.iterrows():
            print(f"    {linha['grupo_1']} e {linha['grupo_2']} "
                  f"(podem ser tratados como um grupo só)")

    if magnitude == "DESPREZÍVEL":
        print(f"\n  Ressalva: apesar de confiável, o efeito é pequeno demais para "
              f"ter uso prático.\n  Com amostra grande, quase tudo dá significativo.")

    _alerta_redundancia(efeito, metrica, variavel, grupo)

    if detalhes:
        print(f"\n  Detalhe técnico:")
        print(f"    Teste global: {teste} | estatística = {estatistica:.4f} | "
              f"{_formatar_p(p_valor)}")
        print(f"    Post-hoc: {nome_posthoc} (diferença = grupo_2 - grupo_1)\n")
        print("    " + tabela.to_string(index=False, float_format=lambda x: f"{x:.4f}")
              .replace("\n", "\n    "))

    return resumo, tabela


# --------------------------------------------------------------------------- #
# Categórica x categórica
# --------------------------------------------------------------------------- #

def comparar_categoricas(df, variavel, grupo, alpha=0.05,
                         ordem_variavel=None, ordem_grupo=None, detalhes=False):
    """Testa a associação entre duas variáveis categóricas (Qui-Quadrado).

    H0: as duas variáveis são independentes.

    Retorna (resumo, tabela_residuos).
    """
    tabela = pd.crosstab(df[variavel], df[grupo])
    if ordem_variavel is not None:
        tabela = tabela.reindex(index=ordem_variavel)
    if ordem_grupo is not None:
        tabela = tabela.reindex(columns=ordem_grupo)

    if tabela.shape[0] < 2 or tabela.shape[1] < 2:
        raise ValueError(f"Cada variável precisa de ao menos 2 categorias. "
                         f"Tabela {tabela.shape[0]}x{tabela.shape[1]}.")

    n = tabela.values.sum()
    _cabecalho(f"{variavel} x {grupo} — tabela {tabela.shape[0]}x{tabela.shape[1]}, n={n}")

    chi2, p_valor, gl, esperado = chi2_contingency(tabela)
    significativo = p_valor < alpha

    # O Qui-Quadrado é uma aproximação: com frequências esperadas baixas ela
    # deixa de valer e o p-valor fica pouco confiável.
    minimo, proporcao_ok = esperado.min(), (esperado >= 5).mean()
    pressuposto_ok = minimo >= 5 and proporcao_ok >= 0.8

    # Cramér's V: o Qui-Quadrado cresce com o N, então sozinho ele não diz
    # se a associação é forte.
    dimensao_minima = min(tabela.shape[0] - 1, tabela.shape[1] - 1)
    cramers_v = np.sqrt(chi2 / (n * dimensao_minima))
    magnitude = _magnitude(cramers_v, "Cramér's V")

    percentuais = tabela.div(tabela.sum(axis=1), axis=0) * 100
    print(f"\n  % de cada {grupo} dentro de cada {variavel}:")
    print("    " + percentuais.round(1).to_string().replace("\n", "\n    "))

    residuos = sm.stats.Table(tabela).standardized_resids
    linhas = [
        {"categoria": categoria, "grupo": coluna,
         "observado": int(tabela.loc[categoria, coluna]),
         "esperado": esperado[i, j], "residuo": residuos.loc[categoria, coluna]}
        for i, categoria in enumerate(tabela.index)
        for j, coluna in enumerate(tabela.columns)
    ]
    tabela_residuos = pd.DataFrame(linhas)
    tabela_residuos["destaque"] = tabela_residuos["residuo"].abs() > LIMIAR_RESIDUO
    tabela_residuos = tabela_residuos.reindex(
        tabela_residuos["residuo"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)
    destaques = tabela_residuos[tabela_residuos["destaque"]]

    # --- conclusão em português ---
    if significativo:
        print(f"\n  '{variavel}' e '{grupo}' estão ASSOCIADAS. "
              f"A associação é {magnitude} (Cramér's V = {cramers_v:.3f}, "
              f"{_formatar_p(p_valor)}).")
        if len(destaques):
            # Só os 3 desvios mais fortes: o resto raramente muda a leitura.
            print(f"\n  Combinações mais fora do esperado:")
            for _, linha in destaques.head(3).iterrows():
                direcao = "mais" if linha["residuo"] > 0 else "menos"
                print(f"    {linha['categoria']} tem {direcao} '{linha['grupo']}' do que "
                      f"o esperado ({linha['observado']} vs {linha['esperado']:.0f})")
            if len(destaques) > 3:
                print(f"    (+{len(destaques) - 3} outras — veja detalhes=True)")
        if magnitude == "DESPREZÍVEL":
            print(f"\n  Ressalva: a associação é confiável mas fraca demais para ter "
                  f"uso prático.")
    else:
        print(f"\n  '{variavel}' e '{grupo}' são INDEPENDENTES ({_formatar_p(p_valor)}).")
        print(f"  Saber uma não ajuda a prever a outra.")

    if not pressuposto_ok:
        alternativa = ("o teste exato de Fisher" if tabela.shape == (2, 2)
                       else "agrupar as categorias raras")
        print(f"\n  ATENÇÃO: há células com poucos casos esperados "
              f"(mínimo {minimo:.1f}).\n  O p-valor é pouco confiável — considere {alternativa}.")

    _alerta_redundancia(cramers_v, "Cramér's V", variavel, grupo)

    if detalhes:
        print(f"\n  Detalhe técnico:")
        print(f"    Qui-Quadrado = {chi2:.4f} | gl = {gl} | {_formatar_p(p_valor)}")
        print(f"    Esperado mínimo = {minimo:.1f} | células com esperado >= 5: "
              f"{proporcao_ok:.0%}")
        print(f"\n    Contagens observadas:")
        print("    " + tabela.to_string().replace("\n", "\n    "))
        print(f"\n    Resíduos padronizados (|r| > {LIMIAR_RESIDUO} destoa):")
        print("    " + tabela_residuos.to_string(
            index=False, float_format=lambda x: f"{x:.2f}").replace("\n", "\n    "))

    resumo = {"variavel": variavel, "grupo": grupo, "teste": "Qui-Quadrado",
              "chi2": chi2, "gl": gl, "p_valor": p_valor,
              "cramers_v": cramers_v, "magnitude": magnitude,
              "esperado_minimo": minimo, "pressuposto_ok": pressuposto_ok,
              "significativo": significativo}
    return resumo, tabela_residuos