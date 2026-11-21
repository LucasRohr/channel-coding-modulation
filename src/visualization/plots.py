from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

from ..core.structures import BitArray, IQSignal, SimulationMetrics


class SimulationPlotter:
    """
    Responsável pela geração e exportação de gráficos analíticos em alta resolução.
    """

    def __init__(self, output_dir: str = "outputs") -> None:
        self.output_path = Path(output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({
            "figure.figsize": (10, 6),
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.6,
            "font.size": 11,
        })

    def plot_ber_curves(
        self,
        curves: Dict[str, List[SimulationMetrics]],
        title: str,
        filename: str,
    ) -> Path:
        """
        Plota curvas BER x Eb/N0 [dB] em escala semilogarítmica, sobrepondo
        limites teóricos quando disponíveis.
        """
        fig, ax = plt.subplots()

        markers = ["o", "s", "^", "D", "v", "P"]
        for idx, (label, metrics_list) in enumerate(curves.items()):
            snr_vals = np.array([m.eb_no_db for m in metrics_list])
            # Evita log(0) substituindo BER=0 por NaN apenas na plotagem logarítmica
            ber_sim = np.array(
                [m.ber_simulated if m.ber_simulated > 0 else np.nan for m in metrics_list]
            )

            marker = markers[idx % len(markers)]
            ax.semilogy(
                snr_vals,
                ber_sim,
                marker=marker,
                linewidth=2,
                markersize=6,
                label=f"{label} (Simulado)",
            )

            if metrics_list[0].ber_theoretical is not None:
                ber_teo = np.array([m.ber_theoretical for m in metrics_list])
                ax.semilogy(
                    snr_vals,
                    ber_teo,
                    linestyle=":",
                    linewidth=1.8,
                    label=f"{label} (Teórico)",
                )

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Relação Energia por Bit / Densidade de Ruído — $E_b/N_0$ [dB]")
        ax.set_ylabel("Taxa de Erro de Bits (BER)")
        ax.set_ylim(1e-5, 1.0)
        ax.legend(loc="lower left", framealpha=0.9)

        file_path = self.output_path / filename
        fig.tight_layout()
        fig.savefig(file_path, dpi=300)
        plt.close(fig)
        return file_path

    def plot_constellations(
        self,
        snapshots: List[Tuple[str, IQSignal, IQSignal]],
        filename: str = "iq_constellations.png",
    ) -> Path:
        """
        Plota uma grade 2x2 de diagramas de constelação I/Q (sinal recebido ruidoso
        sobreposto aos pontos ideais transmitidos).
        """
        fig, axes = plt.subplots(2, 2, figsize=(11, 10))
        axes_flat = axes.flatten()

        for idx, (title, tx_sig, rx_sig) in enumerate(snapshots[:4]):
            ax = axes_flat[idx]
            ax.scatter(
                np.real(rx_sig),
                np.imag(rx_sig),
                alpha=0.35,
                s=12,
                color="#1f77b4",
                label="Símbolos Recebidos (Rx)",
            )
            ideal_points = np.unique(np.round(tx_sig, decimals=4))
            ax.scatter(
                np.real(ideal_points),
                np.imag(ideal_points),
                color="#d62728",
                marker="X",
                s=130,
                edgecolors="black",
                linewidths=1.0,
                zorder=5,
                label="Constelação Ideal (Tx)",
            )
            ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
            ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Componente em Fase — I")
            ax.set_ylabel("Componente em Quadratura — Q")
            ax.set_xlim(-2.2, 2.2)
            ax.set_ylim(-2.2, 2.2)
            ax.legend(loc="upper right", fontsize=9)

        file_path = self.output_path / filename
        fig.tight_layout()
        fig.savefig(file_path, dpi=300)
        plt.close(fig)
        return file_path

    def plot_baseband_waveforms(
        self,
        sample_bits: BitArray,
        manchester_encoded: np.ndarray,
        ami_encoded: np.ndarray,
        filename: str = "baseband_waveforms.png",
    ) -> Path:
        """
        Exibe formas de onda digitais no tempo para comparação visual de transições
        de sincronismo e componente contínua (DC).
        """
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=False)

        # 1. NRZ Polar
        nrz_levels = np.where(sample_bits == 0, 1.0, -1.0)
        t_nrz = np.arange(nrz_levels.size + 1)
        axes[0].step(t_nrz, np.append(nrz_levels, nrz_levels[-1]), where="post", linewidth=2, color="#2ca02c")
        axes[0].set_title(f"Sinalização NRZ Polar — Bits: {''.join(map(str, sample_bits))}", fontweight="bold")
        axes[0].set_ylabel("Tensão [V]")
        axes[0].set_ylim(-1.5, 1.5)

        # 2. Manchester (IEEE 802.3)
        manch_levels = np.where(manchester_encoded == 0, -1.0, 1.0)
        t_manch = np.arange(manch_levels.size + 1) * 0.5
        axes[1].step(t_manch, np.append(manch_levels, manch_levels[-1]), where="post", linewidth=2, color="#1f77b4")
        axes[1].set_title("Codificação Bifásica Manchester (Transição no Meio do Bit)", fontweight="bold")
        axes[1].set_ylabel("Tensão [V]")
        axes[1].set_ylim(-1.5, 1.5)

        # 3. Bipolar AMI
        t_ami = np.arange(ami_encoded.size + 1)
        axes[2].step(t_ami, np.append(ami_encoded, ami_encoded[-1]), where="post", linewidth=2, color="#ff7f0e")
        axes[2].set_title("Codificação Pseudoternária Bipolar AMI (Alternância de Marca)", fontweight="bold")
        axes[2].set_xlabel("Tempo Normalizado ($t / T_b$)")
        axes[2].set_ylabel("Tensão [V]")
        axes[2].set_ylim(-1.5, 1.5)

        file_path = self.output_path / filename
        fig.tight_layout()
        fig.savefig(file_path, dpi=300)
        plt.close(fig)
        return file_path