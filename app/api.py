from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.inference import predict


# ============================================================
# CONFIGURAÇÃO DA API
# ============================================================

app = FastAPI(
    title="Contextual Thompson Sampling API",
    description=(
        "API para recomendação de canal de contato "
        "utilizando Contextual Thompson Sampling."
    ),
    version="1.0.0",
)


# ============================================================
# SCHEMA DE ENTRADA
# ============================================================

class ClientData(BaseModel):

    age: int
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str

    month: str
    day_of_week: str

    campaign: int
    pdays: int
    previous: int
    poutcome: str

    emp_var_rate: float
    cons_price_idx: float
    cons_conf_idx: float
    euribor3m: float
    nr_employed: float

    previous_contact: int
    previous_success: int

    age_group: str
    financial_risk: int
    engagement_score: int


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok"
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
def predict_channel(
    client: ClientData
):

    try:

        # ----------------------------------------------------
        # Converter Pydantic para dict
        # ----------------------------------------------------

        client_data = client.model_dump()

        # ----------------------------------------------------
        # Ajustar nomes para os nomes utilizados no dataset
        # ----------------------------------------------------

        client_data["emp.var.rate"] = (
            client_data.pop("emp_var_rate")
        )

        client_data["cons.price.idx"] = (
            client_data.pop("cons_price_idx")
        )

        client_data["cons.conf.idx"] = (
            client_data.pop("cons_conf_idx")
        )

        client_data["nr.employed"] = (
            client_data.pop("nr_employed")
        )

        # ----------------------------------------------------
        # Inferência
        # ----------------------------------------------------

        result = predict(
            client_data
        )

        return {
            "action": result["action"],
            "score": result["score"],
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )