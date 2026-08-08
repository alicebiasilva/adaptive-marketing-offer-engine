from __future__ import annotations

import numpy as np


class ContextualThompsonSampling:
    """
    Contextual Thompson Sampling usando regressão linear Bayesiana
    para cada arm.

    Cada arm possui seus próprios parâmetros:

        A_arm
        b_arm

    O contexto x representa as características do cliente após
    o pré-processamento/One-Hot Encoding.

    Exemplo de arms:

        {
            "cellular": 0,
            "telephone": 1
        }
    """

    def __init__(
        self,
        n_features: int,
        arms: list[str],
        alpha: float = 1.0,
        random_state: int | None = 42,
    ):
        """
        Parameters
        ----------
        n_features:
            Número de dimensões do vetor de contexto.

        arms:
            Lista de ações disponíveis.

        alpha:
            Parâmetro de regularização/prior.

        random_state:
            Seed para reprodutibilidade.
        """

        if n_features <= 0:
            raise ValueError(
                "n_features deve ser maior que zero."
            )

        if not arms:
            raise ValueError(
                "É necessário informar pelo menos um arm."
            )

        self.n_features = n_features
        self.arms = arms
        self.alpha = alpha

        self.rng = np.random.default_rng(
            random_state
        )

        # ----------------------------------------------------
        # Parâmetros de cada arm
        # ----------------------------------------------------

        self.A = {
            arm: alpha * np.eye(n_features)
            for arm in arms
        }

        self.b = {
            arm: np.zeros(n_features)
            for arm in arms
        }

    # ========================================================
    # PARÂMETROS
    # ========================================================

    def _theta(self, arm: str) -> np.ndarray:
        """
        Calcula o vetor de parâmetros esperado para um arm.
        """

        A_inv = np.linalg.solve(
            self.A[arm],
            np.eye(self.n_features)
        )

        theta = A_inv @ self.b[arm]

        return theta

    # ========================================================
    # AMOSTRAGEM DE THOMPSON
    # ========================================================

    def _sample_theta(self, arm: str) -> np.ndarray:
        """
        Amostra um vetor de parâmetros da distribuição posterior
        do arm.
        """

        A_inv = np.linalg.solve(
            self.A[arm],
            np.eye(self.n_features)
        )

        theta = self._theta(arm)

        sampled_theta = (
            self.rng.multivariate_normal(
                mean=theta,
                cov=A_inv
            )
        )

        return sampled_theta

    # ========================================================
    # PREDIÇÃO / ESCOLHA DA AÇÃO
    # ========================================================

    def select_action(
        self,
        context: np.ndarray,
    ) -> tuple[str, float]:
        """
        Escolhe uma ação utilizando Thompson Sampling.

        Parameters
        ----------
        context:
            Vetor de contexto do cliente.

        Returns
        -------
        action:
            Arm escolhido.

        score:
            Score amostrado utilizado para escolher o arm.
        """

        context = np.asarray(
            context,
            dtype=float
        )

        if context.ndim != 1:
            raise ValueError(
                "context deve ser um vetor 1D."
            )

        if len(context) != self.n_features:
            raise ValueError(
                f"Contexto possui {len(context)} dimensões, "
                f"mas o modelo espera {self.n_features}."
            )

        scores = {}

        for arm in self.arms:

            sampled_theta = (
                self._sample_theta(arm)
            )

            scores[arm] = float(
                context @ sampled_theta
            )

        selected_arm = max(
            scores,
            key=scores.get
        )

        return (
            selected_arm,
            scores[selected_arm]
        )

    # ========================================================
    # ATUALIZAÇÃO DO MODELO
    # ========================================================

    def update(
        self,
        context: np.ndarray,
        arm: str,
        reward: float,
    ) -> None:
        """
        Atualiza o modelo após observar o reward.

        Parameters
        ----------
        context:
            Vetor de características do cliente.

        arm:
            Ação executada.

        reward:
            Reward observado.
            Para seu problema:

                0 = não converteu
                1 = converteu
        """

        context = np.asarray(
            context,
            dtype=float
        )

        if arm not in self.arms:
            raise ValueError(
                f"Arm desconhecido: {arm}"
            )

        if context.ndim != 1:
            raise ValueError(
                "context deve ser um vetor 1D."
            )

        if len(context) != self.n_features:
            raise ValueError(
                f"Contexto possui {len(context)} dimensões, "
                f"mas o modelo espera {self.n_features}."
            )

        # Atualização da matriz A
        self.A[arm] += np.outer(
            context,
            context
        )

        # Atualização do vetor b
        self.b[arm] += (
            reward * context
        )

    # ========================================================
    # TREINAMENTO OFFLINE
    # ========================================================

    def fit(
        self,
        contexts: np.ndarray,
        actions: list[str],
        rewards: np.ndarray,
    ) -> None:
        """
        Inicializa/aprende os parâmetros utilizando histórico.

        Parameters
        ----------
        contexts:
            Matriz:

                (n_observacoes, n_features)

        actions:
            Arm utilizado em cada observação.

        rewards:
            Reward observado em cada observação.
        """

        contexts = np.asarray(
            contexts,
            dtype=float
        )

        rewards = np.asarray(
            rewards,
            dtype=float
        )

        if contexts.ndim != 2:
            raise ValueError(
                "contexts deve ser uma matriz 2D."
            )

        if contexts.shape[1] != self.n_features:
            raise ValueError(
                f"contexts possui "
                f"{contexts.shape[1]} features, "
                f"mas o modelo espera "
                f"{self.n_features}."
            )

        if len(contexts) != len(actions):
            raise ValueError(
                "contexts e actions possuem tamanhos diferentes."
            )

        if len(contexts) != len(rewards):
            raise ValueError(
                "contexts e rewards possuem tamanhos diferentes."
            )

        for context, action, reward in zip(
            contexts,
            actions,
            rewards
        ):
            self.update(
                context=context,
                arm=action,
                reward=reward
            )

    # ========================================================
    # INFORMAÇÕES DO MODELO
    # ========================================================

    def get_parameters(self) -> dict:
        """
        Retorna os parâmetros atuais do modelo.
        """

        return {
            "A": self.A,
            "b": self.b,
        }

    def get_theta(self) -> dict[str, np.ndarray]:
        """
        Retorna os vetores de parâmetros estimados
        para cada arm.
        """

        return {
            arm: self._theta(arm)
            for arm in self.arms
        }