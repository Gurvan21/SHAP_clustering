"""
Découverte causale : apprendre un DAG à partir des données (liens causaux entre variables).
Utilise causal-learn (GES ou PC) pour explorer les dépendances et produire un graphe orienté.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx


def _causal_learn_graph_to_nx(record_G, node_names: list) -> nx.DiGraph:
    """
    Convertit le graphe causal-learn (CPDAG) en NetworkX DiGraph.
    causal-learn: graph[j,i]=1 et graph[i,j]=-1 => i -> j; graph[i,j]=graph[j,i]=-1 => arête non orientée.
    Les arêtes non orientées sont orientées par ordre des nœuds pour obtenir un DAG.
    """
    try:
        g = record_G.graph
    except AttributeError:
        g = np.asarray(record_G)
    n = len(node_names)
    G = nx.DiGraph()
    G.add_nodes_from(node_names)
    used = set()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (i, j) in used or (j, i) in used:
                continue
            # i -> j : dans causal-learn graph[j,i]=1, graph[i,j]=-1
            if g[j, i] == 1 and g[i, j] == -1:
                G.add_edge(node_names[i], node_names[j])
                used.add((i, j))
            elif g[i, j] == -1 and g[j, i] == -1:
                # Arête non orientée : on oriente pour obtenir un DAG (ordre des indices)
                if i < j:
                    G.add_edge(node_names[i], node_names[j])
                else:
                    G.add_edge(node_names[j], node_names[i])
                used.add((i, j))
                used.add((j, i))
    # S'assurer qu'on a un DAG (pas de cycles) : si cycle, enlever la dernière arête qui le crée
    while not nx.is_directed_acyclic_graph(G):
        try:
            cycle = nx.find_cycle(G)
            G.remove_edge(cycle[-1][0], cycle[-1][1])
        except nx.NetworkXNoCycle:
            break
    return G


def discover_dag_ges(df: pd.DataFrame, **kwargs) -> nx.DiGraph:
    """
    Découverte du DAG par GES (Greedy Equivalence Search).
    df : DataFrame avec une colonne par variable (incluant la cible si souhaité).
    Retourne un NetworkX DiGraph avec les noms de colonnes comme nœuds.
    """
    from causallearn.search.ScoreBased.GES import ges
    node_names = list(df.columns)
    X = df.values.astype(np.float64)
    record = ges(X, **kwargs)
    return _causal_learn_graph_to_nx(record["G"], node_names)


def discover_dag_pc(df: pd.DataFrame, alpha: float = 0.05, **kwargs) -> nx.DiGraph:
    """
    Découverte du DAG par l'algorithme PC (tests d'indépendance conditionnelle).
    alpha : seuil pour les tests d'indépendance.
    """
    from causallearn.search.ConstraintBased.PC import pc
    node_names = list(df.columns)
    X = df.values.astype(np.float64)
    record = pc(X, alpha=alpha, **kwargs)
    return _causal_learn_graph_to_nx(record["G"], node_names)


def discover_dag(
    df: pd.DataFrame,
    method: str = "ges",
    **kwargs,
) -> nx.DiGraph:
    """
    Découverte du DAG à partir des données.
    method : "ges" (score-based, BIC) ou "pc" (constraint-based, tests d'indépendance).
    Toutes les colonnes de df sont utilisées pour découvrir les liens causaux.
    """
    method = method.lower()
    if method == "ges":
        return discover_dag_ges(df, **kwargs)
    if method == "pc":
        return discover_dag_pc(df, **kwargs)
    raise ValueError(f"method doit être 'ges' ou 'pc', reçu: {method}")
