from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURAÇÕES
# ============================================================

MODEL_DIR = Path("models")

BANDIT_PATH = MODEL_DIR / "bandit.pkl"
ENCODER_PATH = MODEL_DIR / "encoder.pkl"


# ============================================================
# FEATURES DO CONTEXTO
# ============================================================

CONTEXT_FEATURES = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "month",
    "day_of_week",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
    "previous_contact",
    "previous_success",
    "age_group",
    "financial_risk",
    "engagement_score",
]


# ============================================================
# CARREGAMENTO DOS ARTEFATOS
# ============================================================

def load_models():
    """
    Carrega o Thompson Sampling e o encoder
    previamente treinados.
    """

    if not BANDIT_PATH.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {BANDIT_PATH}"
        )

    if not ENCODER_PATH.exists():
        raise FileNotFoundError(
            f"Encoder não encontrado: {ENCODER_PATH}"
        )

    bandit = joblib.load(BANDIT_PATH)
    encoder = joblib.load(ENCODER_PATH)

    return bandit, encoder


# ============================================================
# PREPARAÇÃO DO CONTEXTO
# ============================================================

def prepare_context(
    client: dict,
    encoder
) -> np.ndarray:
    """
    Recebe os dados de um cliente e transforma
    nas mesmas features utilizadas durante o treinamento.
    """

    df_client = pd.DataFrame([client])

    # Verifica se todas as features estão presentes
    missing_features = [
        feature
        for feature in CONTEXT_FEATURES
        if feature not in df_client.columns
    ]

    if missing_features:
        raise ValueError(
            "Features ausentes: "
            f"{missing_features}"
        )

    # Mantém exatamente a ordem utilizada no treinamento
    df_client = df_client[
        CONTEXT_FEATURES
    ]

    # Mesmo encoder utilizado no treinamento
    encoded = encoder.transform(
        df_client
    )

    encoded = np.asarray(
        encoded,
        dtype=float
    )

    # Adiciona intercepto
    context = np.column_stack(
        [
            np.ones(len(encoded)),
            encoded,
        ]
    )

    return context


# ============================================================
# INFERÊNCIA
# ============================================================

def predict(
    client: dict
) -> dict:
    """
    Recebe um cliente e retorna a ação escolhida
    pelo Thompson Sampling.
    """

    # Carrega os artefatos
    bandit, encoder = load_models()

    # Prepara contexto
    context = prepare_context(
        client,
        encoder
    )

    # Seleciona a ação
    action, score = bandit.select_action(
        context[0]
    )

    return {
        "action": action,
        "score": float(score),
    }


# ============================================================
# TESTE LOCAL
# ============================================================

if __name__ == "__main__":

    client = {
        "age": 40,
        "job": "admin.",
        "marital": "married",
        "education": "university.degree",
        "default": "no",
        "housing": "yes",
        "loan": "no",
        "month": "may",
        "day_of_week": "mon",
        "campaign": 1,
        "pdays": 999,
        "previous": 0,
        "poutcome": "nonexistent",
        "emp.var.rate": 1.1,
        "cons.price.idx": 93.994,
        "cons.conf.idx": -36.4,
        "euribor3m": 4.857,
        "nr.employed": 5191.0,
        "previous_contact": 0,
        "previous_success": 0,
        "age_group": "adult",
        "financial_risk": 0,
        "engagement_score": 1,
    }

    result = predict(client)

    print("=" * 60)
    print("INFERÊNCIA")
    print("=" * 60)

    print(f"Ação escolhida : {result['action']}")
    print(f"Score          : {result['score']:.6f}")