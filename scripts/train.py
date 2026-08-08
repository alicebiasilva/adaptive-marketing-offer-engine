from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from src.bandit import ContextualThompsonSampling


print("=" * 70)
print("PROCESSO INICIADO")
print("=" * 70)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

print("Configurando ambiente...")

RANDOM_STATE = 42
TEST_SIZE = 0.20

DATA_PATH = Path(
    "data/processed/bank_marketing_processed.csv"
)

# Diretório onde os modelos usados pela aplicação serão salvos
MODEL_DIR = Path("models")

BANDIT_PATH = MODEL_DIR / "bandit.pkl"
ENCODER_PATH = MODEL_DIR / "encoder.pkl"


# ============================================================
# MLFLOW
# ============================================================

# O banco SQLite do MLflow ficará separado dos modelos
MLFLOW_DIR = Path("mlflow")

MLFLOW_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MLFLOW_DB = MLFLOW_DIR / "mlflow.db"

# Tracking backend usando SQLite
mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB.resolve()}"
)

# Experimento
mlflow.set_experiment(
    "contextual_thompson_sampling"
)


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
# INÍCIO DO EXPERIMENTO MLFLOW
# ============================================================

with mlflow.start_run() as run:

    print(
        f"\nMLflow Run ID: {run.info.run_id}"
    )

    # ========================================================
    # 1. LEITURA DOS DADOS
    # ========================================================

    print("\nCarregando dados...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(
        f"Dataset: {df.shape}"
    )


    # ========================================================
    # 2. VALIDAÇÃO DAS COLUNAS
    # ========================================================

    print("\nValidando entradas...")

    required_columns = (
        CONTEXT_FEATURES
        + ["contact", "y"]
    )

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Colunas ausentes no dataset: "
            f"{missing_columns}"
        )

    print("Todas as colunas necessárias estão presentes.")


    print("\n")
    print("=" * 70)
    print("INICIANDO TREINAMENTO!")
    print("=" * 70)


    # ========================================================
    # 3. SEPARAÇÃO
    # ========================================================

    X = df[
        CONTEXT_FEATURES
    ].copy()

    actions = df[
        "contact"
    ].copy()

    rewards = df[
        "y"
    ].copy()


    # ========================================================
    # 4. CONVERSÃO DO REWARD
    # ========================================================

    if rewards.dtype == "object":

        reward_mapping = {
            "yes": 1,
            "no": 0,
            "sim": 1,
            "não": 0,
        }

        rewards = rewards.map(
            reward_mapping
        )

    if rewards.isna().any():

        raise ValueError(
            "Existem valores inválidos em y."
        )

    rewards = rewards.astype(float)


    # ========================================================
    # 5. TRAIN / TEST
    # ========================================================

    indices = np.arange(
        len(df)
    )

    (
        train_idx,
        test_idx
    ) = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=rewards,
    )

    X_train = (
        X.iloc[train_idx]
        .copy()
    )

    X_test = (
        X.iloc[test_idx]
        .copy()
    )

    actions_train = (
        actions.iloc[train_idx]
        .reset_index(drop=True)
    )

    actions_test = (
        actions.iloc[test_idx]
        .reset_index(drop=True)
    )

    rewards_train = (
        rewards.iloc[train_idx]
        .reset_index(drop=True)
    )

    rewards_test = (
        rewards.iloc[test_idx]
        .reset_index(drop=True)
    )


    print("\nTreino:")
    print(
        f"X: {X_train.shape}"
    )

    print(
        f"Ações: {actions_train.shape}"
    )

    print(
        f"Rewards: {rewards_train.shape}"
    )


    print("\nTeste:")
    print(
        f"X: {X_test.shape}"
    )

    print(
        f"Ações: {actions_test.shape}"
    )

    print(
        f"Rewards: {rewards_test.shape}"
    )


    # ========================================================
    # 6. IDENTIFICAÇÃO DAS FEATURES
    # ========================================================

    categorical_features = (
        X_train
        .select_dtypes(
            include=["object"]
        )
        .columns
        .tolist()
    )

    numerical_features = [
        col
        for col in CONTEXT_FEATURES
        if col not in categorical_features
    ]

    print("\nFeatures categóricas:")
    print(
        categorical_features
    )

    print("\nFeatures numéricas:")
    print(
        numerical_features
    )


    # ========================================================
    # 7. ONE-HOT ENCODING
    # ========================================================

    encoder = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_features,
            ),
            (
                "numerical",
                "passthrough",
                numerical_features,
            ),
        ]
    )

    X_train_encoded = (
        encoder.fit_transform(
            X_train
        )
    )

    X_test_encoded = (
        encoder.transform(
            X_test
        )
    )

    X_train_encoded = np.asarray(
        X_train_encoded,
        dtype=float
    )

    X_test_encoded = np.asarray(
        X_test_encoded,
        dtype=float
    )


    # ========================================================
    # 8. ADICIONAR INTERCEPT
    # ========================================================

    X_train_context = np.column_stack(
        [
            np.ones(
                len(X_train_encoded)
            ),
            X_train_encoded,
        ]
    )

    X_test_context = np.column_stack(
        [
            np.ones(
                len(X_test_encoded)
            ),
            X_test_encoded,
        ]
    )

    print("\nContexto:")

    print(
        f"Treino: {X_train_context.shape}"
    )

    print(
        f"Teste : {X_test_context.shape}"
    )


    # ========================================================
    # 9. CRIAR O THOMPSON SAMPLING
    # ========================================================

    arms = sorted(
        actions_train.unique()
    )

    print("\nArms:")
    print(arms)

    bandit = ContextualThompsonSampling(
        n_features=X_train_context.shape[1],
        arms=arms,
        alpha=1.0,
        random_state=RANDOM_STATE,
    )


    # ========================================================
    # 10. TREINAMENTO OFFLINE
    # ========================================================

    print("\n" + "=" * 70)
    print("TREINANDO THOMPSON SAMPLING")
    print("=" * 70)

    bandit.fit(
        contexts=X_train_context,
        actions=actions_train.tolist(),
        rewards=rewards_train.to_numpy(),
    )

    print(
        "Treinamento concluído."
    )


    # ========================================================
    # 11. INFORMAÇÕES DOS PARÂMETROS
    # ========================================================

    print("\nParâmetros estimados:")

    theta_dict = bandit.get_theta()

    for arm in arms:

        theta = theta_dict[arm]

        print(
            f"{arm}: "
            f"dimensão={len(theta)}, "
            f"norma={np.linalg.norm(theta):.6f}"
        )


    # ========================================================
    # 12. MÉTRICAS DO TREINAMENTO
    # ========================================================

    train_conversion_rate = (
        rewards_train.mean()
    )

    test_conversion_rate = (
        rewards_test.mean()
    )

    print("\nTaxa de conversão:")

    print(
        f"Treino: {train_conversion_rate:.6f}"
    )

    print(
        f"Teste : {test_conversion_rate:.6f}"
    )


    # ========================================================
    # 13. PARÂMETROS MLFLOW
    # ========================================================

    mlflow.log_param(
        "algorithm",
        "Contextual Thompson Sampling"
    )

    mlflow.log_param(
        "random_state",
        RANDOM_STATE
    )

    mlflow.log_param(
        "test_size",
        TEST_SIZE
    )

    mlflow.log_param(
        "n_original_features",
        len(CONTEXT_FEATURES)
    )

    mlflow.log_param(
        "n_context_features",
        X_train_context.shape[1]
    )

    mlflow.log_param(
        "n_train_samples",
        len(X_train_context)
    )

    mlflow.log_param(
        "n_test_samples",
        len(X_test_context)
    )

    mlflow.log_param(
        "n_arms",
        len(arms)
    )

    mlflow.log_param(
        "arms",
        ",".join(arms)
    )

    mlflow.log_param(
        "alpha",
        1.0
    )

    mlflow.log_param(
        "n_categorical_features",
        len(categorical_features)
    )

    mlflow.log_param(
        "n_numerical_features",
        len(numerical_features)
    )


    # ========================================================
    # 14. MÉTRICAS MLFLOW
    # ========================================================

    mlflow.log_metric(
        "train_conversion_rate",
        float(train_conversion_rate)
    )

    mlflow.log_metric(
        "test_conversion_rate",
        float(test_conversion_rate)
    )

    for arm in arms:

        theta = theta_dict[arm]

        mlflow.log_metric(
            f"{arm}_theta_norm",
            float(
                np.linalg.norm(theta)
            )
        )


    # ========================================================
    # 15. CRIAR DIRETÓRIO DOS MODELOS
    # ========================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # 16. SALVAR BANDIT
    # ========================================================

    joblib.dump(
        bandit,
        BANDIT_PATH
    )

    print(
        f"\nBandit salvo em: "
        f"{BANDIT_PATH}"
    )


    # ========================================================
    # 17. SALVAR ENCODER
    # ========================================================

    joblib.dump(
        encoder,
        ENCODER_PATH
    )

    print(
        f"Encoder salvo em: "
        f"{ENCODER_PATH}"
    )


    # ========================================================
    # 18. REGISTRAR ARTEFATOS NO MLFLOW
    # ========================================================

    print(
        "\nRegistrando artefatos no MLflow..."
    )

    mlflow.log_artifact(
        str(BANDIT_PATH)
    )

    mlflow.log_artifact(
        str(ENCODER_PATH)
    )

    print(
        "Artefatos registrados."
    )


    # ========================================================
    # 19. TESTE RÁPIDO DE INFERÊNCIA
    # ========================================================

    print("\n" + "=" * 70)
    print("TESTE DE INFERÊNCIA")
    print("=" * 70)

    sample_context = (
        X_test_context[0]
    )

    action, score = (
        bandit.select_action(
            sample_context
        )
    )

    print(
        f"Ação selecionada: {action}"
    )

    print(
        f"Score amostrado: {score:.6f}"
    )

    mlflow.log_metric(
        "sample_inference_score",
        float(score)
    )


    # ========================================================
    # 20. RESUMO
    # ========================================================

    print("\n" + "=" * 70)
    print("TREINAMENTO FINALIZADO")
    print("=" * 70)

    print(
        f"Features originais : "
        f"{len(CONTEXT_FEATURES)}"
    )

    print(
        f"Dimensões contexto : "
        f"{X_train_context.shape[1]}"
    )

    print(
        f"Arms               : "
        f"{arms}"
    )

    print(
        f"Observações treino : "
        f"{len(X_train_context)}"
    )

    print(
        f"Bandit             : "
        f"{BANDIT_PATH}"
    )

    print(
        f"Encoder            : "
        f"{ENCODER_PATH}"
    )

    print(
        f"MLflow DB          : "
        f"{MLFLOW_DB}"
    )

    print(
        f"MLflow Run ID      : "
        f"{run.info.run_id}"
    )

    print("=" * 70)