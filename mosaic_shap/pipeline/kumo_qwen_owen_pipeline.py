"""
Pipeline Owen/Winter enrichi — version définitive.

BUGS CORRIGÉS dans ce fichier :
────────────────────────────────
BUG 1 — Qwen3 : AttributeError: 'NoneType' object has no attribute 'strip'
  Cause : Qwen3 active le mode "thinking" par défaut → message.content = None.
  Fix   : extra_body={"enable_thinking": False} + méthode _extract_content()
          qui lit reasoning_content en fallback.

BUG 2 — KumoRFM : pydantic ValidationError (Input should be a valid string)
  Cause : rfm.predict() n'accepte QUE une chaîne PQL, jamais un DataFrame ou
          un array numpy. La syntaxe PQL interdit aussi les crochets [0,1,2].
  Fix   : on construit toujours une chaîne PQL avec des parenthèses (0,1,2,...).
          Pour SHAP KernelExplainer (qui appelle predict avec un numpy array),
          on utilise le fallback GradientBoosting au lieu de KumoRFM — SHAP
          n'est pas compatible avec le mode PQL de KumoRFM directement.
"""

import warnings
import numpy as np
import pandas as pd
import json
import os
import sys
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import hdbscan




from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import shap

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
try:
    from mosaic_shap.explain.Owen_Shap.owen_explainer import OWENExplainer
    from mosaic_shap.explain.Winter_Shap.winter_explainer import WINTERExplainer
    MOSAIC_OK = True
    print("OWENExplainer défini")
except ImportError:
    MOSAIC_OK = False
    print("⚠ mosaic_shap non trouvé — Owen/Winter approximés")


# ══════════════════════════════════════════════════════════════════════════════
# PARTIE 1 — KumoRFM-2
# ══════════════════════════════════════════════════════════════════════════════

class KumoRFMWrapper:
    """
    Wrapper KumoRFM-2.

    ARCHITECTURE DE predict() :
    ───────────────────────────
    KumoRFM.predict() n'accepte QUE une chaîne PQL.
    Il est impossible de lui passer un numpy array ou un DataFrame directement.

    Pour SHAP (qui appelle predict avec un array numpy), on utilise donc
    le modèle fallback (GradientBoosting) comme proxy de f(x).
    KumoRFM est utilisé pour ses GROUPES (structure de graphe) pas pour SHAP.

    Méthodes exposées :
    - predict_pql(ids)  : prédictions KumoRFM via PQL pour une liste d'IDs
    - predict(X)        : prédictions via fallback (pour SHAP)
    - get_graph_based_groups(X) : groupes Owen/Winter depuis la structure de graphe
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key      = api_key or os.environ.get("KUMO_API_KEY", "")
        self._rfm         = None
        self._graph       = None
        self._fallback    = None   # GradientBoosting — utilisé pour SHAP
        self._entity_col  = "entity_id"
        self._target_col  = "target"

    def build_graph_from_single_table(
        self, X: pd.DataFrame, y: pd.Series, entity_col: str = "entity_id"
    ) -> bool:
        self._entity_col = entity_col
        try:
            from kumoai.experimental.rfm import Graph, KumoRFM

            X_r = X.reset_index(drop=True).copy()
            y_r = y.reset_index(drop=True).copy()

            geo_cols   = [c for c in X_r.columns
                          if c in ['Latitude', 'Longitude', 'Population', 'AveOccup']]
            hous_cols  = [c for c in X_r.columns if c not in geo_cols]
            shared_key = "district_id"

            districts_df = X_r[geo_cols].copy()
            districts_df[shared_key] = range(len(districts_df))

            housing_df = X_r[hous_cols].copy()
            housing_df[entity_col]       = range(len(housing_df))
            housing_df[shared_key]       = range(len(housing_df))
            housing_df[self._target_col] = y_r.values

            # Graphe KumoRFM — FK inférée automatiquement via le nom partagé
            self._graph = Graph.from_data({
                "districts": districts_df,
                "housing":   housing_df,
            })
            self._graph["districts"].primary_key = shared_key
            self._graph["housing"].primary_key   = entity_col

            self._rfm = KumoRFM(self._graph)

            # ── Fallback GradientBoosting pour SHAP ───────────────────────────
            # KumoRFM ne peut pas être utilisé directement par SHAP (PQL only).
            # On entraîne un GB sur les mêmes données comme proxy explicatif.
            self._fallback = GradientBoostingRegressor(
                n_estimators=100, random_state=42
            )
            self._fallback.fit(X_r, y_r)

            print("✅ KumoRFM-2 initialisé et prêt")
            return True

        except Exception as e:
            print(f"⚠ Erreur KumoRFM : {e} — fallback GradientBoosting seul")
            self._fallback = GradientBoostingRegressor(
                n_estimators=200, random_state=42
            )
            self._fallback.fit(X, y)
            return False

    def predict_pql(self, ids: List[int]) -> np.ndarray:
        """
        BUG 2 CORRIGÉ : syntaxe PQL avec parenthèses, pas crochets.

        PQL valide   : FOR housing.entity_id IN (0, 1, 2)
        PQL invalide : FOR housing.entity_id IN [0, 1, 2]   ← erreur originale
        """
        if self._rfm is None:
            raise RuntimeError("KumoRFM non initialisé.")

        # Syntaxe tuple SQL — pas de crochets Python
        ids_str = "(" + ", ".join(str(i) for i in ids) + ")"
        query = (
            f"PREDICT AVG(housing.{self._target_col}, 0, 365, days) "
            f"FOR housing.{self._entity_col} IN {ids_str}"
        )
        result = self._rfm.predict(query)
        if isinstance(result, pd.DataFrame):
            return result['prediction'].values
        return np.array(result).ravel()

    def predict(self, X) -> np.ndarray:
        """
        Interface sklearn-compatible pour SHAP et Owen/Winter.
        Utilise le fallback GradientBoosting (pas KumoRFM PQL).
        """
        if self._fallback is not None:
            return self._fallback.predict(X)
        raise RuntimeError("Fallback non disponible.")

    def get_graph_based_groups(
        self, X: pd.DataFrame
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """Groupes Owen/Winter depuis la structure de graphe KumoRFM."""
        feature_names = list(X.columns)
        geo_cols  = ['Latitude', 'Longitude', 'Population', 'AveOccup']
        geo_idx   = [i for i, c in enumerate(feature_names) if c in geo_cols]
        hous_idx  = [i for i, c in enumerate(feature_names) if c not in geo_cols]
        coarse_groups = [g for g in [hous_idx, geo_idx] if g]

        fine_groups = []
        for cg in coarse_groups:
            if len(cg) <= 2:
                fine_groups.append(cg)
            else:
                sub  = X.iloc[:, cg]
                corr = sub.corr().abs()
                dist = (1 - corr.values + (1 - corr.values).T) / 2
                np.fill_diagonal(dist, 0)
                from scipy.spatial.distance import squareform
                Z = linkage(squareform(dist), method='ward')
                sub_labels = fcluster(Z, t=min(2, len(cg)//2), criterion='maxclust')
                for k in np.unique(sub_labels):
                    idx = [cg[j] for j in np.where(sub_labels == k)[0]]
                    if idx:
                        fine_groups.append(idx)

        print(f"✅ Groupes KumoRFM : {len(coarse_groups)} grossiers, "
              f"{len(fine_groups)} fins")
        return coarse_groups, fine_groups


# ══════════════════════════════════════════════════════════════════════════════
# PARTIE 2 — Qwen3
# ══════════════════════════════════════════════════════════════════════════════

class Qwen3GroupingOracle:
    """
    BUG 1 CORRIGÉ : mode thinking Qwen3
    ─────────────────────────────────────
    Qwen3 active le thinking par défaut → message.content = None.
    Fix : extra_body={"enable_thinking": False} dans la requête API
    + méthode _extract_content() qui lit reasoning_content en fallback.
    """

    MODELS = [
        #"Qwen/Qwen3-235B-A22B",
        "Qwen/Qwen2.5-72B-Instruct",
        "microsoft/Phi-4",
        "HuggingFaceH4/zephyr-7b-beta",
        "microsoft/Phi-3-mini-4k-instruct",
        "Qwen/Qwen2-7B-Instruct",
    ]

    SYSTEM_PROMPT = """Tu es un expert XAI. Analyse les features et propose une hiérarchie
Owen/Winter. Tu DOIS répondre UNIQUEMENT par un objet JSON valide. 
Commence immédiatement par `{` et termine par `}`. 
N'ajoute AUCUN texte avant ou après, ni aucun commentaire. 

Format de réponse :
{
  "coarse_groups": {"nom_1": ["feat1","feat2"], "nom_2": ["feat3","feat4"]},
  "fine_groups":   {"nom_A": ["feat1"], "nom_B": ["feat2"], "nom_C": ["feat3","feat4"]},
  "justification": "...",
  "causal_hint":   "..."
}
Règles : chaque feature dans exactement 1 groupe grossier ET 1 groupe fin.
Les groupes fins sont des sous-ensembles des groupes grossiers. 


"""

    def __init__(self, mode: str = "api", hf_token: Optional[str] = None):
        self.mode      = mode
        self.hf_token  = hf_token or os.environ.get("HF_TOKEN", "")
        self._pipeline = None

    # ── BUG 1 CORRIGÉ ─────────────────────────────────────────────────────────
    def _extract_content(self, message) -> str:
        """
        Extrait le texte d'une réponse Qwen3 en gérant le mode thinking.

        En mode thinking actif :
          message.content           = None   ← AttributeError dans l'ancien code
          message.reasoning_content = "..."  ← contient le raisonnement

        On désactive le thinking avec enable_thinking=False, mais on garde
        ce fallback en cas de comportement inattendu.
        """
        # Cas normal : content rempli (thinking désactivé)
        if message.content is not None:
            return str(message.content).strip()

        # Fallback 1 : attribut reasoning_content (Qwen3 spécifique)
        if hasattr(message, 'reasoning_content') and message.reasoning_content:
            return str(message.reasoning_content).strip()

        # Fallback 2 : model_extra (dict des champs non-standard OpenAI)
        if hasattr(message, 'model_extra') and message.model_extra:
            for key in ['reasoning_content', 'thinking', 'reasoning']:
                val = message.model_extra.get(key)
                if val:
                    return str(val).strip()

        return ""

    def _build_prompt(
        self,
        feature_names: List[str],
        shap_means: np.ndarray,
        corr_matrix: np.ndarray,
        domain_description: str = ""
    ) -> str:
        lines = [
            f"Dataset : {domain_description or 'données tabulaires'}",
            "Importance SHAP moyenne par feature :",
        ]
        for f, v in zip(feature_names, shap_means):
            lines.append(f"  {f}: {v:.4f}")
        lines.append("Corrélations SHAP fortes (|r| > 0.5) :")
        found = False
        M = len(feature_names)
        for i in range(M):
            for j in range(i+1, M):
                if abs(corr_matrix[i, j]) > 0.5:
                    lines.append(
                        f"  {feature_names[i]} ↔ {feature_names[j]}: "
                        f"{corr_matrix[i,j]:.2f}"
                    )
                    found = True
        if not found:
            lines.append("  (aucune corrélation forte)")
        lines.append(f"\nPropose la hiérarchie Owen/Winter pour ces {M} features.")
        return "\n".join(lines)

    def get_groups_via_api(
        self,
        feature_names: List[str],
        shap_means: np.ndarray,
        corr_matrix: np.ndarray,
        domain_description: str = ""
    ) -> Dict:
        from huggingface_hub import InferenceClient

        # provider="hf-inference" → endpoint HF natif (pas "auto" qui échoue
        # avec les clés HF standards car il tente des providers tiers)
        client = InferenceClient(
            #provider="hf-inference",
            api_key=self.hf_token,
        )
        user_prompt = self._build_prompt(
            feature_names, shap_means, corr_matrix, domain_description
        )
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]

        last_error = None
        for model_id in self.MODELS:
            try:
                print(f"  → Tentative {model_id}...", flush=True)
                response = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    max_tokens=800,
                    temperature=0.3,
                    # BUG 1 CORRIGÉ : désactive le thinking → content non-None
                    extra_body={"enable_thinking": False},
                )
                raw = self._extract_content(response.choices[0].message)
                if raw:
                    return self._parse_json_response(raw, feature_names)
                print("     ⚠ Réponse vide, modèle suivant...")
            except Exception as e:
                last_error = e
                print(f"     ⚠ Erreur ({e}), modèle suivant...")

        print(f"⚠ Tous les modèles HF ont échoué ({last_error}) — fallback")
        return self._fallback_groups(feature_names)

    def get_groups_via_local(
        self,
        feature_names: List[str],
        shap_means: np.ndarray,
        corr_matrix: np.ndarray,
        domain_description: str = ""
    ) -> Dict:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import torch

        if self._pipeline is None:
            model_id = "Qwen/Qwen3-8B"
            print(f"Chargement {model_id} (peut prendre quelques minutes)...")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model_obj = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_4bit=True
            )
            self._pipeline = pipeline(
                "text-generation", model=model_obj, tokenizer=tokenizer,
                max_new_tokens=512, do_sample=False, temperature=None,
            )

        user_prompt = self._build_prompt(
            feature_names, shap_means, corr_matrix, domain_description
        )
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]
        tokenizer = self._pipeline.tokenizer
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            # BUG 1 CORRIGÉ : désactive le thinking en local aussi
            enable_thinking=False,
        )
        outputs = self._pipeline(text)
        raw = outputs[0]["generated_text"].split("<|im_start|>assistant")[-1].strip()
        return self._parse_json_response(raw, feature_names)

    def _parse_json_response(self, raw: str, feature_names: List[str]) -> Dict:
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        print("Réponse brute:", repr(raw), start, end)
        if start == -1 or end == 0:
            print("⚠ Pas de JSON dans la réponse — fallback")
            return self._fallback_groups(feature_names)
        try:
            data = json.loads(raw[start:end])
        except json.JSONDecodeError as e:
            print(f"⚠ JSON invalide ({e}) — fallback")
            return self._fallback_groups(feature_names)

        fn_lower = {f.lower(): i for i, f in enumerate(feature_names)}

        def names_to_indices(group_dict):
            result = []
            for _, feats in group_dict.items():
                idx_list = []
                for f in feats:
                    key = f.lower()
                    if key in fn_lower:
                        idx_list.append(fn_lower[key])
                    else:
                        matches = [k for k in fn_lower if key in k or k in key]
                        if matches:
                            idx_list.append(fn_lower[matches[0]])
                if idx_list:
                    result.append(idx_list)
            return result

        coarse_groups = names_to_indices(data.get("coarse_groups", {}))
        fine_groups   = names_to_indices(data.get("fine_groups",   {}))

        all_idx    = set(range(len(feature_names)))
        missing_c  = all_idx - {i for g in coarse_groups for i in g}
        missing_f  = all_idx - {i for g in fine_groups   for i in g}
        if missing_c:
            coarse_groups.append(list(missing_c))
        if missing_f:
            fine_groups.append(list(missing_f))

        return {
            "coarse_groups": coarse_groups,
            "fine_groups":   fine_groups,
            "justification": data.get("justification", ""),
            "causal_hint":   data.get("causal_hint",   ""),
        }

    def _fallback_groups(self, feature_names: List[str]) -> Dict:
        M, h = len(feature_names), len(feature_names) // 2
        return {
            "coarse_groups": [list(range(h)), list(range(h, M))],
            "fine_groups":   [list(range(h//2)), list(range(h//2, h)),
                              list(range(h, h+(M-h)//2)),
                              list(range(h+(M-h)//2, M))],
            "justification": "Groupes égaux par défaut (LLM indisponible).",
            "causal_hint":   "",
        }

    def get_groups(
        self,
        feature_names: List[str],
        shap_means: np.ndarray,
        corr_matrix: np.ndarray,
        domain_description: str = ""
    ) -> Dict:
        if self.mode == "api":
            return self.get_groups_via_api(
                feature_names, shap_means, corr_matrix, domain_description
            )
        else:
            return self.get_groups_via_local(
                feature_names, shap_means, corr_matrix, domain_description
            )


# ══════════════════════════════════════════════════════════════════════════════
# PARTIE 3 — Pipeline complet
# ══════════════════════════════════════════════════════════════════════════════

def run_enhanced_pipeline(
    use_kumo:  bool = False,
    use_qwen:  bool = False,
    qwen_mode: str  = "api",
    n_sample:  int  = 200,
    n_perm:    int  = 16,
):
    print("=" * 60)
    print("PIPELINE ENRICHI : KumoRFM-2 + Qwen3 + Owen/Winter")
    print("=" * 60)

    # ── Données ───────────────────────────────────────────────────────────────
    data = fetch_california_housing(as_frame=True)
    idx  = data.data.sample(n=n_sample, random_state=42).index
    X    = data.data.loc[idx].reset_index(drop=True)
    y    = data.target.loc[idx].reset_index(drop=True)
    fn   = list(X.columns)
    M    = len(fn)
    print(f"N={n_sample}, M={M} features")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    X_tr = X_tr.reset_index(drop=True)
    X_te = X_te.reset_index(drop=True)
    y_tr = y_tr.reset_index(drop=True)
    y_te = y_te.reset_index(drop=True)
    background = X_tr.values[:50]

    # ── Étape 1 : Modèle ──────────────────────────────────────────────────────
    print("\n── ÉTAPE 1 : Modèle de prédiction ──")
    if use_kumo:
        print("Utilisation de KumoRFM-2...")
        kumo = KumoRFMWrapper(api_key=os.environ.get("KUMO_API_KEY"))
        kumo.build_graph_from_single_table(X_tr, y_tr)
        model = kumo
        # Le modèle SHAP est le fallback GB (KumoRFM n'est pas compatible numpy)
        shap_model = kumo._fallback
    else:
        print("RandomForest (n_jobs=1)...")
        model      = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
        model.fit(X_tr, y_tr)
        shap_model = model
        print(f"R² test = {r2_score(y_te, model.predict(X_te)):.3f}")

    # ── Étape 2 : SHAP ────────────────────────────────────────────────────────
    print("\n── ÉTAPE 2 : SHAP ordre 1 ──")
    # BUG 2 CORRIGÉ : on utilise toujours shap_model (sklearn-compatible),
    # jamais KumoRFMWrapper directement avec KernelExplainer
    explainer = shap.TreeExplainer(shap_model)
    shap_vals  = explainer.shap_values(X_te.values)
    X_eval     = X_te.values

    shap_means = np.abs(shap_vals).mean(0)
    shap_corr  = np.corrcoef(shap_vals.T)
    print(f"SHAP shape : {shap_vals.shape}  |  "
          f"Feature la + importante : {fn[np.argmax(shap_means)]}")

    # ── Étape 3 : Groupes ─────────────────────────────────────────────────────
    print("\n── ÉTAPE 3 : Découverte des groupes ──")
    coarse_groups = fine_groups = group_method = None

    if use_kumo and isinstance(model, KumoRFMWrapper) and model._graph is not None:
        coarse_groups, fine_groups = model.get_graph_based_groups(X_tr)
        group_method = "KumoRFM-2 (structure de graphe)"

    elif use_qwen:
        oracle = Qwen3GroupingOracle(mode=qwen_mode, hf_token=os.environ.get("HF_TOKEN"))
        result = oracle.get_groups(
            feature_names=fn,
            shap_means=shap_means,
            corr_matrix=shap_corr,
            domain_description="Prix médian maisons Californie (8 features géo-socio-éco)"
        )
        coarse_groups = result["coarse_groups"]
        fine_groups   = result["fine_groups"]
        group_method  = "Qwen3 (sémantique LLM)"
        print(f"  Justification : {result['justification']}")
        print(f"  Indice causal : {result['causal_hint']}")

    # Fallback si aucune méthode n'a produit de groupes
    if not coarse_groups:
        Z      = linkage(pdist(shap_vals.T, metric='correlation'), method='ward')
        lbl_c  = fcluster(Z, t=2, criterion='maxclust')
        lbl_f  = fcluster(Z, t=4, criterion='maxclust')
        coarse_groups = [np.where(lbl_c == k)[0].tolist() for k in [1, 2]]
        fine_groups   = [np.where(lbl_f == k)[0].tolist()
                         for k in range(1, 5) if (lbl_f == k).any()]
        group_method  = "Clustering SHAP (baseline)"

    print(f"\nMéthode : {group_method}")
    for k, cg in enumerate(coarse_groups):
        print(f"  GC{k} : {[fn[i] for i in cg]}")
    for k, fg in enumerate(fine_groups):
        print(f"  GF{k} : {[fn[i] for i in fg]}")

    # ── Étape 4 : Owen & Winter ───────────────────────────────────────────────
    print(f"\n── ÉTAPE 4 : Owen & Winter ──")
    N_EX = min(20, len(X_eval))
    Xe   = X_eval[:N_EX]
    sv   = shap_vals[:N_EX]

    # Owen et Winter utilisent shap_model (sklearn-compatible)
    if MOSAIC_OK and shap_model is not None:
        print(f"Calcul Owen  (N={N_EX}, n_perm={n_perm})...")
        owen_vals = OWENExplainer(
            model=shap_model, groups=fine_groups,
            background=background, n_permutations=n_perm
        ).shap_values(Xe)

        print(f"Calcul Winter (N={N_EX}, n_perm={n_perm})...")
        winter_vals = WINTERExplainer(
            model=shap_model,
            coarse_groups=coarse_groups, fine_groups=fine_groups,
            background=background, n_permutations=n_perm
        ).shap_values(Xe)
    else:
        print("⚠ Owen/Winter approximés (mosaic_shap indisponible)")
        owen_vals   = sv.copy()
        winter_vals = sv.copy()

    # ── Étape 5 : Divergence ──────────────────────────────────────────────────
    print("\n── ÉTAPE 5 : Divergence Owen–Shapley ──")
    div   = np.abs(owen_vals - sv).mean(0)
    order = np.argsort(div)[::-1]
    dmax  = div.max() + 1e-8
    for i in order:
        bar = "█" * max(1, int(div[i] / dmax * 20))
        print(f"  {fn[i]:15s} | {bar:<20s} | {div[i]:.4f}")
    print(f"\n→ '{fn[order[0]]}' : divergence maximale (effet structuré par le groupe)")

    return {
        "shap_values":   sv,
        "owen_values":   owen_vals,
        "winter_values": winter_vals,
        "coarse_groups": coarse_groups,
        "fine_groups":   fine_groups,
        "feature_names": fn,
        "group_method":  group_method,
        "shap_model":    shap_model,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PARTIE 4 — Qwen3 comme explainer des clusters Owen/Winter
# ══════════════════════════════════════════════════════════════════════════════

def explain_clusters_with_qwen3(
    owen_vals: np.ndarray,
    cluster_labels: np.ndarray,
    feature_names: List[str],
    hf_token: Optional[str] = None,
    mode: str = "api"
) -> Dict[int, str]:
    oracle = Qwen3GroupingOracle(mode=mode, hf_token=hf_token)
    descriptions = {}

    for cid in np.unique(cluster_labels):
        mask = (cluster_labels == cid)
        cluster_mean = np.abs(owen_vals[mask]).mean(0)
        top3 = np.argsort(cluster_mean)[::-1][:3]

        prompt = (
            f"Cluster {cid} ({mask.sum()} observations).\n\n"
            f"Top-3 features Owen les plus importantes :\n"
            f"1. {feature_names[top3[0]]} : {cluster_mean[top3[0]]:.4f}\n"
            f"2. {feature_names[top3[1]]} : {cluster_mean[top3[1]]:.4f}\n"
            f"3. {feature_names[top3[2]]} : {cluster_mean[top3[2]]:.4f}\n\n"
            f"En une phrase, explique ce que représente ce cluster pour le modèle "
            f"de prédiction du prix des maisons en Californie."
        )
        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(
                #provider="hf-inference",
                api_key=hf_token or os.environ.get("HF_TOKEN", "")
            )
            resp = client.chat.completions.create(
                model="Qwen/Qwen2.5-72B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150, temperature=0.4,
                extra_body={"enable_thinking": False},
            )
            # BUG 1 CORRIGÉ ici aussi
            desc = oracle._extract_content(resp.choices[0].message)
        except Exception as e:
            desc = (f"Cluster dominé par {feature_names[top3[0]]}, "
                    f"{feature_names[top3[1]]}, {feature_names[top3[2]]}. "
                    f"(LLM indisponible : {e})")

        descriptions[int(cid)] = desc
        print(f"\nCluster {cid} : {desc}")

    return descriptions


# =============================================================================
# 1. Algorithme de triple clustering consensus
# =============================================================================

def triple_clustering_consensus(X, n_clusters_range=(2, 5), random_state=42):
    """
    Applique trois algorithmes de clustering (KMeans, HDBSCAN, Agglomerative)
    et fusionne leurs résultats par vote majoritaire pour produire un consensus.

    Paramètres
    ----------
    X : array-like
        Matrice des données à clusteriser (ex: valeurs Owen ou SHAP).
    n_clusters_range : tuple
        Plage de recherche pour le nombre de clusters (min, max).
    random_state : int
        Graine aléatoire pour la reproductibilité.

    Retourne
    --------
    dict
        Dictionnaire contenant les labels de chaque algorithme et les labels consensus.
    """
    # 1. KMeans (meilleur silhouette sur la plage)
    best_kmeans = None
    best_labels_k = None
    best_score_k = -1
    for k in range(n_clusters_range[0], n_clusters_range[1] + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        if len(set(labels)) > 1:
            score = silhouette_score(X, labels)
            if score > best_score_k:
                best_score_k = score
                best_labels_k = labels
                best_kmeans = kmeans

    # 2. HDBSCAN (recherche automatique de min_cluster_size)
    best_hdb = None
    best_labels_h = None
    best_score_h = -1
    for min_size in [3, 5, 7, 10]:
        hdb = hdbscan.HDBSCAN(min_cluster_size=min_size, min_samples=1)
        labels = hdb.fit_predict(X)
        n_clust = len(set(labels)) - (1 if -1 in labels else 0)
        if 2 <= n_clust <= 5:
            mask = labels != -1
            if mask.sum() > 1 and len(set(labels[mask])) > 1:
                score = silhouette_score(X[mask], labels[mask])
                if score > best_score_h:
                    best_score_h = score
                    best_labels_h = labels
                    best_hdb = hdb
    if best_labels_h is None:
        best_labels_h = np.zeros(len(X), dtype=int)  # fallback

    # 3. Agglomerative Clustering (meilleur silhouette)
    best_agg = None
    best_labels_a = None
    best_score_a = -1
    for k in range(n_clusters_range[0], n_clusters_range[1] + 1):
        agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = agg.fit_predict(X)
        if len(set(labels)) > 1:
            score = silhouette_score(X, labels)
            if score > best_score_a:
                best_score_a = score
                best_labels_a = labels
                best_agg = agg

    # 4. Consensus par vote majoritaire
    consensus_labels = np.full(len(X), -1, dtype=int)
    for i in range(len(X)):
        votes = [best_labels_k[i], best_labels_h[i], best_labels_a[i]]
        valid_votes = [v for v in votes if v != -1]
        if len(valid_votes) >= 2:
            from collections import Counter
            counter = Counter(valid_votes)
            most_common, count = counter.most_common(1)[0]
            if count >= 2:
                consensus_labels[i] = most_common

    # Renumérotation des clusters consensus
    unique = np.unique(consensus_labels[consensus_labels != -1])
    mapping = {old: new for new, old in enumerate(unique)}
    consensus_labels = np.array([mapping.get(x, -1) if x != -1 else -1 for x in consensus_labels])

    return {
        'kmeans_labels': best_labels_k,
        'hdbscan_labels': best_labels_h,
        'agg_labels': best_labels_a,
        'consensus_labels': consensus_labels,
        'kmeans_model': best_kmeans,
        'hdbscan_model': best_hdb,
        'agg_model': best_agg
    }


# =============================================================================
# 2. Fonction dyadique (valeur + explication)
# =============================================================================

def dyadic_triple_clustering(owen_vals, shap_vals, feature_names,
                             n_clusters_range=(2,5), random_state=42):
    """
    Applique le triple clustering consensus séparément dans l'espace des valeurs
    Owen et dans l'espace des explications SHAP, puis construit une partition
    dyadique (concaténation des deux labels consensus).
    """
    # Clustering sur les valeurs Owen
    owen_res = triple_clustering_consensus(owen_vals, n_clusters_range, random_state)
    labels_owen = owen_res['consensus_labels']

    # Clustering sur les valeurs SHAP
    shap_res = triple_clustering_consensus(shap_vals, n_clusters_range, random_state)
    labels_shap = shap_res['consensus_labels']

    # Labels dyadiques (paire de clusters)
    dyad_labels = [f"O{lo}_S{ls}" for lo, ls in zip(labels_owen, labels_shap)]

    # Création d'un DataFrame récapitulatif
    df = pd.DataFrame(shap_vals, columns=feature_names)
    df['cluster_owen'] = labels_owen
    df['cluster_shap'] = labels_shap
    df['dyad'] = dyad_labels

    return {
        'labels_owen': labels_owen,
        'labels_shap': labels_shap,
        'dyad_labels': dyad_labels,
        'df': df,
        'owen_clustering': owen_res,
        'shap_clustering': shap_res
    }


# =============================================================================
# 3. Visualisation et analyse des résultats
# =============================================================================

def visualize_dyadic_clustering(results, feature_names, save_figures=False):
    """
    Visualise les clusters consensus (Owen, SHAP, dyadique) sous forme de :
    - Heatmap des profils moyens par cluster
    - Barplots des features les plus discriminantes
    - Matrice de confusion entre clusters Owen et SHAP
    - Dendrogramme des clusters consensus
    """
    owen_labels = results['labels_owen']
    shap_labels = results['labels_shap']
    dyad_labels = results['dyad_labels']
    df = results['df']

    # 1. Heatmap des profils moyens par cluster Owen
    owen_profiles = df.groupby('cluster_owen')[feature_names].mean()
    plt.figure(figsize=(10,6))
    sns.heatmap(owen_profiles, annot=True, fmt='.3f', cmap='RdBu_r', center=0)
    plt.title("Profils moyens des clusters Owen")
    plt.ylabel("Cluster Owen")
    plt.xlabel("Features")
    if save_figures:
        plt.savefig("figures/cluster_owen_profiles.png", dpi=150)
    else:
        plt.show()

    # 2. Heatmap des profils moyens par cluster SHAP
    shap_profiles = df.groupby('cluster_shap')[feature_names].mean()
    plt.figure(figsize=(10,6))
    sns.heatmap(shap_profiles, annot=True, fmt='.3f', cmap='RdBu_r', center=0)
    plt.title("Profils moyens des clusters SHAP")
    plt.ylabel("Cluster SHAP")
    plt.xlabel("Features")
    if save_figures:
        plt.savefig("figures/cluster_shap_profiles.png", dpi=150)
    else:
        plt.show()

    # 3. Matrice de confusion Owen vs SHAP
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(owen_labels, shap_labels)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Clusters SHAP")
    plt.ylabel("Clusters Owen")
    plt.title("Correspondance entre clusters Owen et SHAP")
    if save_figures:
        plt.savefig("figures/confusion_owen_shap.png", dpi=150)
    else:
        plt.show()

    # 4. Barplots des features les plus discriminantes (Owen)
    var_inter = ((owen_profiles - owen_profiles.mean())**2).sum(axis=0).sort_values(ascending=False)
    top_features = var_inter.head(6).index
    fig, axes = plt.subplots(1, len(top_features), figsize=(15,4), sharey=True)
    if len(top_features) == 1:
        axes = [axes]
    for ax, feat in zip(axes, top_features):
        values = [owen_profiles.loc[c, feat] for c in sorted(owen_profiles.index)]
        ax.bar(range(len(values)), values, color='skyblue')
        ax.set_title(feat)
        ax.set_xlabel("Cluster Owen")
        ax.set_ylabel("Valeur moyenne")
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([f"C{c}" for c in sorted(owen_profiles.index)])
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    plt.suptitle("Features les plus discriminantes entre clusters Owen")
    plt.tight_layout()
    if save_figures:
        plt.savefig("figures/owen_discriminant_features.png", dpi=150)
    else:
        plt.show()

    # 5. Dendrogramme des clusters consensus (Owen)
    from scipy.cluster.hierarchy import dendrogram, linkage
    linkage_matrix = linkage(owen_profiles, method='ward')
    plt.figure(figsize=(10,6))
    dendrogram(linkage_matrix, labels=owen_profiles.index, leaf_rotation=90)
    plt.title("Dendrogramme des clusters Owen")
    plt.ylabel("Distance")
    if save_figures:
        plt.savefig("figures/dendrogram_owen.png", dpi=150)
    else:
        plt.show()

    # 6. Tableau récapitulatif des effectifs par dyade
    print("\n=== Effectifs des dyades (cluster_owen, cluster_shap) ===")
    print(pd.crosstab(owen_labels, shap_labels))


# =============================================================================
# 4. Intégration dans le pipeline principal
# =============================================================================

def run_dyadic_clustering_after_pipeline(results, n_clusters_range=(2,5), save_figures=False):
    """
    À exécuter après run_enhanced_pipeline pour réaliser le clustering dyadique.
    """
    owen_vals = results['owen_values']
    shap_vals = results['shap_values']
    feature_names = results['feature_names']

    # Clustering dyadique
    dyadic = dyadic_triple_clustering(owen_vals, shap_vals, feature_names,
                                      n_clusters_range=n_clusters_range)

    # Visualisation
    visualize_dyadic_clustering(dyadic, feature_names, save_figures=save_figures)

    return dyadic

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE  # optionnel

def plot_cluster_scatter(owen_vals, labels, title="Clusters", method='pca', save_path=None):
    """
    Réduit la dimension (PCA ou t-SNE) et affiche les points colorés par cluster.
    labels : array des labels (ex: cluster_owen, cluster_shap, ou dyad)
    """
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        coords = reducer.fit_transform(owen_vals)
        method_name = "PCA"
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(owen_vals)-1))
        coords = reducer.fit_transform(owen_vals)
        method_name = "t-SNE"
    else:
        raise ValueError("method must be 'pca' or 'tsne'")
    
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='tab10', alpha=0.7, edgecolors='k')
    plt.colorbar(scatter, label="Cluster ID")
    plt.title(f"{title} - {method_name} projection")
    plt.xlabel("Composante 1")
    plt.ylabel("Composante 2")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline Owen/Winter enrichi")
    parser.add_argument("--kumo",      action="store_true")
    parser.add_argument("--qwen",      action="store_true")
    parser.add_argument("--qwen-mode", default="api", choices=["api", "local"])
    parser.add_argument("--explain",   action="store_true")
    parser.add_argument("--n",         type=int, default=200)
    parser.add_argument("--n-perm",    type=int, default=16)
    args = parser.parse_args()

    results = run_enhanced_pipeline(
        use_kumo  = args.kumo,
        use_qwen  = args.qwen,
        qwen_mode = args.qwen_mode,
        n_sample  = args.n,
        n_perm    = args.n_perm,
    )

    # Clustering dyadique et visualisations (heatmaps, confusion, dendrogramme)
    dyadic_results = run_dyadic_clustering_after_pipeline(
        results,
        n_clusters_range=(2,5),
        save_figures=True   # sauvegarde les figures dans le dossier figures/
    )

    # Extraction des valeurs pour les scatter plots
    owen_vals = results['owen_values']
    shap_vals = results['shap_values']
    
    # Affichage des clusters Owen en 2D (PCA)
    plot_cluster_scatter(
        owen_vals, 
        dyadic_results['labels_owen'], 
        title="Clusters consensus Owen", 
        method='pca',
        save_path="figures/owen_clusters_pca.png"
    )
    #plt.show()  # ← important pour afficher la figure

    # Affichage des clusters SHAP
    plot_cluster_scatter(
        shap_vals, 
        dyadic_results['labels_shap'], 
        title="Clusters consensus SHAP", 
        method='pca',
        save_path="figures/shap_clusters_pca.png"
    )
    #plt.show()

    # Pour les dyades (labels textuels) : encodage en entiers
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    dyad_labels_encoded = le.fit_transform(dyadic_results['dyad_labels'])
    plot_cluster_scatter(
        owen_vals, 
        dyad_labels_encoded, 
        title="Dyades (Owen_SHAP)", 
        method='pca',
        save_path="figures/dyad_clusters_pca.png"
    )
    #plt.show()

    # Descriptions LLM des clusters (optionnel)
    if args.explain and args.qwen:
        from sklearn.cluster import KMeans
        cluster_labels = KMeans(n_clusters=3, random_state=42).fit_predict(
            results["owen_values"]
        )
        print("\n── ÉTAPE 6 : Descriptions LLM des clusters Owen ──")
        explain_clusters_with_qwen3(
            owen_vals=results["owen_values"],
            cluster_labels=cluster_labels,
            feature_names=results["feature_names"],
            hf_token=os.environ.get("HF_TOKEN"),
            mode=args.qwen_mode
        )

    print("\n" + "=" * 60)
    print("Terminé.")
    print("\nUsage :")
    print("  python kumo_qwen_owen_pipeline.py                 # baseline")
    print("  python kumo_qwen_owen_pipeline.py --kumo          # KumoRFM-2")
    print("  python kumo_qwen_owen_pipeline.py --qwen          # Qwen3 (HF_TOKEN)")
    print("  python kumo_qwen_owen_pipeline.py --kumo --qwen   # les deux")
    print("  python kumo_qwen_owen_pipeline.py --qwen --explain # + LLM clusters")
    


"""
############################################################
# Utilisation générique de Qwen3 pour nos futurs datasets
############################################################

import shap
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Calculer les SHAP values (sur un échantillon)
explainer = shap.TreeExplainer(model)  # ou KernelExplainer pour d'autres modèles
X_sample = X.sample(100, random_state=42)  # sous-ensemble pour rapidité
shap_vals = explainer.shap_values(X_sample)

# 2. Moyenne des valeurs absolues par feature
shap_means = np.abs(shap_vals).mean(axis=0)

# 3. Matrice de corrélation des SHAP values
shap_corr = np.corrcoef(shap_vals.T)

# 4. Appel à l'oracle
oracle = Qwen3GroupingOracle(mode="api", hf_token=os.environ.get("HF_TOKEN"))
result = oracle.get_groups(
    feature_names=list(X.columns),
    shap_means=shap_means,
    corr_matrix=shap_corr,
    domain_description="Description de votre dataset (ex: prédiction de prix, classification, etc.)"
)

coarse_groups = result["coarse_groups"]   # list of lists of indices
fine_groups = result["fine_groups"]   




######################################################################
# Groupes fins et grossiers en utilisant KUMORFM2
######################################################################

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

class GenericKumoWrapper(KumoRFMWrapper):
    
    Version générique de KumoRFMWrapper qui ne dépend pas de noms de colonnes spécifiques.
    
    def build_graph_from_dataframe(
        self, X: pd.DataFrame, y: pd.Series = None, n_tables: int = 2
    ) -> bool:
        
        Construit automatiquement un graphe KumoRFM-2 à partir d'un seul DataFrame.
        - Divise les colonnes en `n_tables` groupes via clustering hiérarchique.
        - Crée une clé étrangère commune (entity_id) entre ces tables.
        - Si y est fourni, il est ajouté dans la première table.
        
        try:
            from kumoai.experimental.rfm import Graph, KumoRFM
        except ImportError:
            print("⚠ kumoai non installé")
            return False

        # 1. Clustering des colonnes pour les répartir en plusieurs tables
        n_cols = X.shape[1]
        if n_cols <= n_tables:
            # Pas assez de colonnes : on met tout dans une seule table
            col_groups = [list(range(n_cols))] * n_tables
        else:
            # Clustering hiérarchique basé sur la corrélation des colonnes
            corr = X.corr().abs().values
            dist = 1 - corr
            np.fill_diagonal(dist, 0)
            Z = linkage(pdist(dist), method='ward')
            labels = fcluster(Z, t=n_tables, criterion='maxclust')
            col_groups = [np.where(labels == k)[0].tolist() for k in range(1, n_tables+1)]

        # 2. Création des tables avec une clé commune
        entity_col = "entity_id"
        shared_key = "shared_id"
        X_with_id = X.copy()
        X_with_id[entity_col] = range(len(X))
        
        tables = {}
        for i, cols in enumerate(col_groups):
            table_name = f"table_{i}"
            df = X_with_id[[entity_col] + [X.columns[c] for c in cols]].copy()
            df[shared_key] = range(len(df))
            if i == 0 and y is not None:
                df['target'] = y.values
            tables[table_name] = df
        
        # 3. Construction du graphe
        self._graph = Graph.from_data(tables)
        # Définition des clés primaires
        for name in tables.keys():
            self._graph[name].primary_key = shared_key
        
        self._rfm = KumoRFM(self._graph)
        
        # Fallback GradientBoosting pour SHAP
        self._fallback = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self._fallback.fit(X, y if y is not None else np.zeros(len(X)))
        
        print(f"✅ Graphe générique KumoRFM construit avec {n_tables} tables")
        return True

    def get_generic_groups(self, X: pd.DataFrame, shap_vals: np.ndarray = None) -> Tuple[List[List[int]], List[List[int]]]:
        
        Retourne les groupes grossiers (tables) et fins (clustering intra-table).
        Si shap_vals est fourni, le clustering fin utilise les corrélations des SHAP,
        sinon il utilise les corrélations des données brutes.
        
        if self._graph is None:
            raise RuntimeError("Graph non construit. Appelez build_graph_from_dataframe d'abord.")
        
        # Groupes grossiers : chaque table du graphe
        table_names = list(self._graph.tables.keys())
        feature_names = list(X.columns)
        # Pour chaque table, récupérer les indices des colonnes correspondantes
        coarse_groups = []
        for tname in table_names:
            cols_in_table = [c for c in feature_names if c in self._graph[tname].df.columns]
            idx = [feature_names.index(c) for c in cols_in_table if c in feature_names]
            if idx:
                coarse_groups.append(idx)
        
        # Groupes fins : clustering intra-table basé sur les corrélations
        fine_groups = []
        for cg in coarse_groups:
            if len(cg) <= 2:
                fine_groups.append(cg)
                continue
            # Utiliser les SHAP values si disponibles, sinon les données brutes
            if shap_vals is not None:
                sub = shap_vals[:, cg]
                corr = np.corrcoef(sub.T)
            else:
                sub = X.iloc[:, cg]
                corr = sub.corr().abs().values
            dist = 1 - np.abs(corr)
            np.fill_diagonal(dist, 0)
            Z = linkage(pdist(dist), method='ward')
            n_sub = min(3, len(cg)//2 + 1)  # nombre de sous-groupes
            sub_labels = fcluster(Z, t=n_sub, criterion='maxclust')
            for k in np.unique(sub_labels):
                idx = [cg[j] for j in np.where(sub_labels == k)[0]]
                if idx:
                    fine_groups.append(idx)
        
        print(f" Groupes génériques : {len(coarse_groups)} grossiers, {len(fine_groups)} fins")
        return coarse_groups, fine_groups


     # Au lieu de l'appel spécifique à California Housing
kumo = GenericKumoWrapper(api_key=os.environ.get("KUMO_API_KEY"))
kumo.build_graph_from_dataframe(X_train, y_train, n_tables=2)

# Le modèle pour SHAP est le fallback (GradientBoosting)
shap_model = kumo._fallback   # ou kumo.predict (qui utilise aussi le fallback)

# Calcul des SHAP values (comme dans votre pipeline)
explainer = shap.TreeExplainer(shap_model)   # ou KernelExplainer selon le modèle
shap_vals = explainer.shap_values(X_test) ou shap.TreeExplainer(kumo._fallback).shap_values(X_test)
# Obtention des groupes génériques
coarse_groups, fine_groups = kumo.get_generic_groups(X_train, shap_vals=shap_vals)        
        
    
    
    
"""