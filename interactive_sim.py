#!/usr/bin/env python3
"""Interactive SIR model simulation with adjustable parameters."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

from sir_model import SIRParameters, run_sir_simulation


def _make_slider(
    ax: plt.Axes,
    label: str,
    valmin: float,
    valmax: float,
    valinit: float,
    valstep: float | None = None,
) -> Slider:
    return Slider(
        ax=ax,
        label=label,
        valmin=valmin,
        valmax=valmax,
        valinit=valinit,
        valstep=valstep,
    )


def main() -> None:
    fig = plt.figure(figsize=(11, 8))
    fig.canvas.manager.set_window_title("SIR Infection Simulation")

    plot_ax = fig.add_axes([0.08, 0.34, 0.86, 0.60])
    plot_ax.set_xlabel("Time")
    plot_ax.set_ylabel("Population")
    plot_ax.set_title("SIR Compartment Model")
    plot_ax.grid(True, alpha=0.3)

    (line_s,) = plot_ax.plot([], [], color="#2563eb", linewidth=2, label="Susceptible (S)")
    (line_i,) = plot_ax.plot([], [], color="#dc2626", linewidth=2, label="Infected (I)")
    (line_r,) = plot_ax.plot([], [], color="#16a34a", linewidth=2, label="Recovered (R)")
    plot_ax.legend(loc="upper right")

    stats_text = plot_ax.text(
        0.02,
        0.98,
        "",
        transform=plot_ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    slider_axes = {
        "N": fig.add_axes([0.12, 0.24, 0.76, 0.025]),
        "beta": fig.add_axes([0.12, 0.20, 0.76, 0.025]),
        "gamma": fig.add_axes([0.12, 0.16, 0.76, 0.025]),
        "S0": fig.add_axes([0.12, 0.11, 0.76, 0.025]),
        "I0": fig.add_axes([0.12, 0.07, 0.76, 0.025]),
        "R0": fig.add_axes([0.12, 0.03, 0.76, 0.025]),
    }

    sliders = {
        "N": _make_slider(slider_axes["N"], "N (total population)", 100, 10000, 1000, valstep=100),
        "beta": _make_slider(slider_axes["beta"], "β (transmission rate)", 0.0, 1.0, 0.3, valstep=0.01),
        "gamma": _make_slider(slider_axes["gamma"], "γ (recovery rate)", 0.01, 1.0, 0.1, valstep=0.01),
        "S0": _make_slider(slider_axes["S0"], "S₀ (initial susceptible)", 0, 10000, 990, valstep=10),
        "I0": _make_slider(slider_axes["I0"], "I₀ (initial infected)", 0, 1000, 10, valstep=1),
        "R0": _make_slider(slider_axes["R0"], "R₀ (initial recovered)", 0, 1000, 0, valstep=1),
    }

    reset_ax = fig.add_axes([0.82, 0.005, 0.12, 0.04])
    reset_button = Button(reset_ax, "Reset", hovercolor="#e5e7eb")

    defaults = {name: slider.val for name, slider in sliders.items()}

    def update(_: object | None = None) -> None:
        try:
            params = SIRParameters(
                N=sliders["N"].val,
                beta=sliders["beta"].val,
                gamma=sliders["gamma"].val,
                S0=sliders["S0"].val,
                I0=sliders["I0"].val,
                R0=sliders["R0"].val,
            )
            result = run_sir_simulation(params)
        except ValueError as exc:
            stats_text.set_text(f"Invalid parameters:\n{exc}")
            line_s.set_data([], [])
            line_i.set_data([], [])
            line_r.set_data([], [])
            fig.canvas.draw_idle()
            return

        line_s.set_data(result.t, result.S)
        line_i.set_data(result.t, result.I)
        line_r.set_data(result.t, result.R)

        plot_ax.relim()
        plot_ax.autoscale_view()

        S0, I0, R0 = params.normalized_initial_conditions()
        stats_text.set_text(
            "\n".join(
                [
                    f"ℛ₀ = β/γ ≈ {params.basic_reproduction_number:.2f}",
                    f"R_eff(0) ≈ {params.effective_reproduction_number:.2f}",
                    f"Peak infected: {result.peak_infected:.0f} at t = {result.peak_time:.1f}",
                    f"Initial (normalized): S={S0:.0f}, I={I0:.0f}, R={R0:.0f}",
                    f"Final: S={result.S[-1]:.0f}, I={result.I[-1]:.0f}, R={result.R[-1]:.0f}",
                ]
            )
        )
        fig.canvas.draw_idle()

    def reset(_: object) -> None:
        for name, value in defaults.items():
            sliders[name].set_val(value)
        update()

    for slider in sliders.values():
        slider.on_changed(update)

    reset_button.on_clicked(reset)
    update()
    plt.show()


if __name__ == "__main__":
    main()
