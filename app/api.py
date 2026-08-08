from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.inference import predict, update


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
# SCHEMA DE ENTRADA DO CLIENTE
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
# SCHEMA DO UPDATE
# ============================================================

class UpdateRequest(BaseModel):

    client: ClientData

    action: str

    reward: int


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok"
    }


# ============================================================
# FUNÇÃO AUXILIAR
# ============================================================

def prepare_client_data(
    client: ClientData
) -> dict:
    """
    Converte o objeto Pydantic para dict e ajusta
    os nomes das colunas para os nomes utilizados
    no dataset e no encoder.
    """

    client_data = client.model_dump()

    # --------------------------------------------------------
    # Variáveis com nomes diferentes no JSON/API
    # --------------------------------------------------------

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

    return client_data


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
def predict_channel(
    client: ClientData
):
    """
    Recebe os dados de um cliente e utiliza
    Thompson Sampling para escolher o canal.
    """

    try:

        # ----------------------------------------------------
        # Preparar dados
        # ----------------------------------------------------

        client_data = prepare_client_data(
            client
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


# ============================================================
# UPDATE / APRENDIZADO ONLINE
# ============================================================

@app.post("/update")
def update_model(
    request: UpdateRequest
):
    """
    Atualiza o Thompson Sampling após observar
    o resultado da interação com o cliente.

    reward:
        0 = não converteu
        1 = converteu
    """

    try:

        # ----------------------------------------------------
        # Validar reward
        # ----------------------------------------------------

        if request.reward not in (0, 1):

            raise HTTPException(
                status_code=400,
                detail="reward deve ser 0 ou 1."
            )

        # ----------------------------------------------------
        # Preparar dados do cliente
        # ----------------------------------------------------

        client_data = prepare_client_data(
            request.client
        )

        # ----------------------------------------------------
        # Atualizar modelo
        # ----------------------------------------------------

        result = update(
            client=client_data,
            action=request.action,
            reward=request.reward
        )

        return {
            "status": result["status"],
            "action": result["action"],
            "reward": result["reward"],
            "model_path": result["model_path"],
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )