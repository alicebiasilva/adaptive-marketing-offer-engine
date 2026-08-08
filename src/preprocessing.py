from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURAÇÕES
# ============================================================

RAW_DATA_PATH = Path(
    "data/raw/bank-additional-full.csv"
)

PROCESSED_DATA_PATH = Path(
    "data/processed/bank_marketing_processed.csv"
)


# ============================================================
# 1. LEITURA DOS DADOS
# ============================================================

def load_data(
    path: Path = RAW_DATA_PATH
) -> pd.DataFrame:

    df = pd.read_csv(
        path,
        sep=";",
        na_values=["unknown"]
    )

    return df


# ============================================================
# 2. TRATAMENTO DA VARIÁVEL TARGET
# ============================================================

def prepare_target(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    df["y"] = (
        df["y"]
        .map({
            "yes": 1,
            "no": 0
        })
    )

    return df


# ============================================================
# 3. FEATURE: PREVIOUS CONTACT
# ============================================================

def create_previous_contact(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    df["previous_contact"] = (
        (df["pdays"] != 999)
        .astype(int)
    )

    return df


# ============================================================
# 4. FEATURE: PREVIOUS SUCCESS
# ============================================================

def create_previous_success(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    df["previous_success"] = (
        (df["poutcome"] == "success")
        .astype(int)
    )

    return df


# ============================================================
# 5. FEATURE: AGE GROUP
# ============================================================

def create_age_group(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    bins = [
        0,
        25,
        35,
        50,
        65,
        float("inf")
    ]

    labels = [
        "young",
        "young_adult",
        "adult",
        "senior",
        "elderly"
    ]

    df["age_group"] = pd.cut(
        df["age"],
        bins=bins,
        labels=labels,
        right=False
    )

    return df


# ============================================================
# 6. FEATURE: FINANCIAL RISK
# ============================================================

def create_financial_risk(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    df["financial_risk"] = (
        (df["default"] == "yes").astype(int)
        + (df["housing"] == "yes").astype(int)
        + (df["loan"] == "yes").astype(int)
    )

    return df


# ============================================================
# 7. FEATURE: ENGAGEMENT SCORE
# ============================================================

def create_engagement_score(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    df["engagement_score"] = (
        df["previous_success"]
        + df["previous_contact"]
        + (df["campaign"] <= 2).astype(int)
        + (df["previous"] > 0).astype(int)
    )

    return df


# ============================================================
# 8. TRATAMENTO DE VALORES AUSENTES
# ============================================================

def handle_missing_values(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    categorical_columns = (
        df
        .select_dtypes(
            include=["object", "category"]
        )
        .columns
    )

    for column in categorical_columns:

        df[column] = (
            df[column]
            .astype("object")
            .fillna("unknown")
        )

    return df


# ============================================================
# 9. REMOÇÃO DE DURATION
# ============================================================

def remove_duration(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    if "duration" in df.columns:

        df = df.drop(
            columns=["duration"]
        )

    return df


# ============================================================
# 10. PIPELINE DE PREPROCESSING
# ============================================================

def preprocess(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = prepare_target(df)

    df = create_previous_contact(df)

    df = create_previous_success(df)

    df = create_age_group(df)

    df = create_financial_risk(df)

    df = create_engagement_score(df)

    df = handle_missing_values(df)

    df = remove_duration(df)

    return df


# ============================================================
# 11. EXECUÇÃO
# ============================================================

def main():

    print("=" * 70)
    print("PREPROCESSAMENTO")
    print("=" * 70)

    print(
        f"\nLendo dados de:\n"
        f"{RAW_DATA_PATH}"
    )

    df = load_data()

    print(
        f"Dataset bruto: {df.shape}"
    )

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    df_processed = preprocess(df)

    print(
        f"Dataset processado: "
        f"{df_processed.shape}"
    )

    # --------------------------------------------------------
    # Criar diretório
    # --------------------------------------------------------

    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Salvar
    # --------------------------------------------------------

    df_processed.to_csv(
        PROCESSED_DATA_PATH,
        index=False
    )

    print(
        f"\nArquivo salvo em:\n"
        f"{PROCESSED_DATA_PATH}"
    )

    print("\nColunas finais:")
    print(
        df_processed.columns.tolist()
    )


if __name__ == "__main__":
    main()