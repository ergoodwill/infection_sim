"""SIR compartment model for infectious disease spread."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class SIRParameters:
    N: float
    beta: float
    gamma: float
    S0: float
    I0: float
    R0: float
    t_max: float = 160.0
    num_points: int = 1000

    def __post_init__(self) -> None:
        if self.N <= 0:
            raise ValueError("Population size N must be positive.")
        if self.beta < 0 or self.gamma < 0:
            raise ValueError("Transmission and recovery rates must be non-negative.")
        if any(v < 0 for v in (self.S0, self.I0, self.R0)):
            raise ValueError("Initial compartment sizes must be non-negative.")
        if self.S0 + self.I0 + self.R0 <= 0:
            raise ValueError("At least one individual must be present at t=0.")

    @property
    def initial_total(self) -> float:
        return self.S0 + self.I0 + self.R0

    @property
    def basic_reproduction_number(self) -> float:
        """Basic reproduction number R0 = beta / gamma (fully susceptible population)."""
        return self.beta / self.gamma if self.gamma > 0 else float("inf")

    @property
    def effective_reproduction_number(self) -> float:
        """Effective reproduction number at t=0: (beta / gamma) * (S0 / N)."""
        return self.basic_reproduction_number * (self.S0 / self.N)

    def normalized_initial_conditions(self) -> tuple[float, float, float]:
        """Scale S0, I0, R0 so they sum exactly to N."""
        total = self.initial_total
        scale = self.N / total
        return self.S0 * scale, self.I0 * scale, self.R0 * scale


@dataclass(frozen=True)
class SIRResult:
    t: np.ndarray
    S: np.ndarray
    I: np.ndarray
    R: np.ndarray
    params: SIRParameters

    @property
    def peak_infected(self) -> float:
        return float(np.max(self.I))

    @property
    def peak_time(self) -> float:
        return float(self.t[np.argmax(self.I)])


def _sir_derivatives(t: float, y: np.ndarray, beta: float, gamma: float, N: float) -> list[float]:
    S, I, R = y
    infection = beta * S * I / N
    dS = -infection
    dI = infection - gamma * I
    dR = gamma * I
    return [dS, dI, dR]


def run_sir_simulation(params: SIRParameters) -> SIRResult:
    """Integrate the SIR ODEs over time."""
    S0, I0, R0 = params.normalized_initial_conditions()
    y0 = [S0, I0, R0]
    t_eval = np.linspace(0.0, params.t_max, params.num_points)

    solution = solve_ivp(
        _sir_derivatives,
        (0.0, params.t_max),
        y0,
        args=(params.beta, params.gamma, params.N),
        t_eval=t_eval,
        method="RK45",
        dense_output=False,
    )

    if not solution.success:
        raise RuntimeError(f"Integration failed: {solution.message}")

    return SIRResult(
        t=solution.t,
        S=solution.y[0],
        I=solution.y[1],
        R=solution.y[2],
        params=params,
    )
