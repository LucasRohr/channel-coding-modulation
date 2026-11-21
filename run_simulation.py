from pathlib import Path
from typing import Dict, List
import numpy as np

from src.coding.ami import AMICodec
from src.coding.hamming import Hamming74Codec
from src.coding.manchester import ManchesterCodec
from src.core.channel import AWGNChannel, ImpulsiveBurstChannel
from src.core.source import DataStream
from src.core.structures import SimulationMetrics
from src.modulation.bpsk import BPSKModulator
from src.modulation.qpsk import QPSKModulator
from src.simulation.engine import MonteCarloSimulator, ScenarioConfig
from src.simulation.metrics import PerformanceAnalyzer
from src.visualization.plots import SimulationPlotter


def main() -> None:
    log_lines: List[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        log_lines.append(msg)

    log("=" * 78)
    log("SIMULADOR DE CODIFICACAO DE CANAL E MODULACAO DIGITAL - REDES II")
    log("=" * 78)

    # Configuracoes iniciais do simulador e plotador de graficos
    simulator = MonteCarloSimulator(seed=2026)
    plotter = SimulationPlotter(output_dir="outputs")

    # Canais de comunicacao: AWGN puro e canal com rajadas impulsivas
    awgn = AWGNChannel(seed=2026)
    burst_channel = ImpulsiveBurstChannel(
        burst_probability=0.008,
        burst_length=6,
        impulse_amplitude=4.5,
        seed=2026,
    )

    # Moduladores e codificadores de canal a serem testados
    bpsk = BPSKModulator()
    qpsk = QPSKModulator()
    hamming = Hamming74Codec()
    manchester = ManchesterCodec()
    ami = AMICodec()

    # -------------------------------------------------------------------------
    # ETAPA 1: DEMONSTRACAO COM MENSAGEM ASCII REAL
    # -------------------------------------------------------------------------
    ascii_msg = "UNISINOS 2026 - Redes de Computadores II: Camada Fisica e Enlace"
    raw_bits = DataStream.text_to_bits(ascii_msg)

    log("\n[ETAPA 1] TRANSMISSAO DE MENSAGEM ASCII")
    log(f"Mensagem Original : '{ascii_msg}'")
    log(f"Tamanho em Bytes  : {len(ascii_msg)} bytes ({raw_bits.size} bits)")
    log(f"Primeiros 32 bits : {''.join(map(str, raw_bits[:32]))}...")

    demo_scenarios = [
        ScenarioConfig("BPSK Nao Codificado", None, bpsk, awgn),
        ScenarioConfig("QPSK Nao Codificado", None, qpsk, awgn),
        ScenarioConfig("Hamming(7,4) + QPSK", hamming, qpsk, awgn),
        ScenarioConfig("AMI Bipolar (Banda Base)", ami, None, awgn),
    ]

    for snr_test in [3.0, 7.0, 11.0]:
        log(f"\n--- Teste ASCII sob Eb/N0 = {snr_test:.1f} dB ---")
        for sc in demo_scenarios:
            res = simulator.simulate_ascii_message(ascii_msg, sc, eb_no_db=snr_test)
            log(
                f"  {sc.name:<26} | Erros: {res.bit_errors:3d}/{res.total_bits} "
                f"| BER: {res.ber:.5f} | Rx: '{res.recovered_text[:42]}...'"
            )

    # -------------------------------------------------------------------------
    # ETAPA 2: SIMULACAO MONTE CARLO (BER x Eb/N0)
    # -------------------------------------------------------------------------
    log("\n[ETAPA 2] SIMULACAO MONTE CARLO (BER x Eb/N0)")
    eb_no_range = np.arange(0.0, 12.0, 1.5)

    sc_bpsk = ScenarioConfig("BPSK", None, bpsk, awgn)
    sc_qpsk = ScenarioConfig("QPSK", None, qpsk, awgn)
    sc_hamming_bpsk = ScenarioConfig("Hamming(7,4) + BPSK", hamming, bpsk, awgn)
    sc_hamming_qpsk = ScenarioConfig("Hamming(7,4) + QPSK", hamming, qpsk, awgn)
    sc_manchester_bpsk = ScenarioConfig("Manchester + BPSK", manchester, bpsk, awgn)
    sc_ami_baseband = ScenarioConfig("AMI Bipolar (Banda Base)", ami, None, awgn)
    sc_hamming_burst = ScenarioConfig("Hamming(7,4) + BPSK (Rajada)", hamming, bpsk, burst_channel)

    all_scenarios = [
        sc_bpsk,
        sc_qpsk,
        sc_hamming_bpsk,
        sc_hamming_qpsk,
        sc_manchester_bpsk,
        sc_ami_baseband,
        sc_hamming_burst,
    ]

    # Executa a varredura Monte Carlo para todos os cenários e armazena os resultados
    results_map: Dict[str, List[SimulationMetrics]] = {}
    for sc in all_scenarios:
        log(f"Executando varredura Monte Carlo: {sc.name}...")
        metrics_list = simulator.run_ber_curve(sc, eb_no_range)
        results_map[sc.name] = metrics_list

    # Tabela resumida de BER
    header = f"{'Eb/N0 [dB]':>10} | " + " | ".join(f"{name[:14]:>14}" for name in results_map)
    log("\n" + header)
    log("-" * len(header))
    for i, snr in enumerate(eb_no_range):
        row_vals = [f"{results_map[name][i].ber_simulated:14.6f}" for name in results_map]
        log(f"{snr:10.1f} | " + " | ".join(row_vals))

    # -------------------------------------------------------------------------
    # ETAPA 3: EFICIENCIA ESPECTRAL E LIMITE DE SHANNON (B = 1 MHz)
    # -------------------------------------------------------------------------
    log("\n[ETAPA 3] ANALISE DE EFICIENCIA ESPECTRAL E CAPACIDADE (B = 1 MHz)")
    spectral_configs = [
        ("BPSK Nao Codificado", 1, 1.0),
        ("QPSK Nao Codificado (Gray)", 2, 1.0),
        ("Hamming(7,4) + BPSK", 1, 4.0 / 7.0),
        ("Hamming(7,4) + QPSK", 2, 4.0 / 7.0),
        ("Manchester + BPSK", 1, 0.5),
        ("AMI Bipolar (Banda Base)", 1, 1.0),
    ]

    spec_header = (
        f"{'Esquema':<28} | {'k (bits/simb)':>13} | {'Taxa Rc':>8} | "
        f"{'Ef. Nom [bps/Hz]':>16} | {'Ef. Efet [bps/Hz]':>17} | "
        f"{'Taxa Util [Mbps]':>16} | {'Shannon Min Eb/N0':>17}"
    )
    log(spec_header)
    log("-" * len(spec_header))

    for name, k, rc in spectral_configs:
        rep = PerformanceAnalyzer.compute_spectral_report(name, k, rc, bandwidth_hz=1e6)
        log(
            f"{rep.scheme_name:<28} | {rep.bits_per_symbol:13d} | {rep.code_rate:8.3f} | "
            f"{rep.nominal_efficiency_bps_hz:16.2f} | {rep.effective_efficiency_bps_hz:17.3f} | "
            f"{rep.net_bitrate_at_1mhz_mbps:16.3f} | {rep.shannon_min_eb_no_db:14.2f} dB"
        )

    # -------------------------------------------------------------------------
    # ETAPA 4: EXPORTACAO DE GRAFICOS E LOGS
    # -------------------------------------------------------------------------
    log("\n[ETAPA 4] GERACAO DE GRAFICOS ANALITICOS")

    p1 = plotter.plot_ber_curves(
        curves={"BPSK": results_map["BPSK"], "QPSK": results_map["QPSK"]},
        title="Desempenho Comparativo de Modulação Digital em Canal AWGN (BPSK vs. QPSK)",
        filename="01_ber_vs_snr_modulations.png",
    )
    log(f" -> Gráfico salvo: {p1}")

    p2 = plotter.plot_ber_curves(
        curves={
            "BPSK Puro (AWGN)": results_map["BPSK"],
            "Hamming(7,4) + BPSK (AWGN)": results_map["Hamming(7,4) + BPSK"],
            "Manchester + BPSK (AWGN)": results_map["Manchester + BPSK"],
            "AMI Bipolar (Banda Base)": results_map["AMI Bipolar (Banda Base)"],
            "Hamming(7,4) + BPSK (Rajada)": results_map["Hamming(7,4) + BPSK (Rajada)"],
        },
        title="Impacto da Codificação de Canal e Ruído em Rajada na BER",
        filename="02_ber_vs_snr_coding_impact.png",
    )
    log(f" -> Gráfico salvo: {p2}")

    snapshots = [
        ("BPSK em Canal AWGN (Eb/N0 = 4 dB)", *simulator.capture_constellation_snapshot(bpsk, awgn, 4.0)),
        ("BPSK em Canal AWGN (Eb/N0 = 12 dB)", *simulator.capture_constellation_snapshot(bpsk, awgn, 12.0)),
        ("QPSK Gray em Canal AWGN (Eb/N0 = 4 dB)", *simulator.capture_constellation_snapshot(qpsk, awgn, 4.0)),
        ("QPSK Gray em Canal AWGN (Eb/N0 = 12 dB)", *simulator.capture_constellation_snapshot(qpsk, awgn, 12.0)),
    ]
    p3 = plotter.plot_constellations(snapshots, filename="03_iq_constellations.png")
    log(f" -> Gráfico salvo: {p3}")

    sample_16_bits = raw_bits[:16]
    p4 = plotter.plot_baseband_waveforms(
        sample_bits=sample_16_bits,
        manchester_encoded=manchester.encode(sample_16_bits),
        ami_encoded=ami.encode(sample_16_bits),
        filename="04_baseband_waveforms.png",
    )
    log(f" -> Gráfico salvo: {p4}")

    log_file = Path("outputs") / "simulation_log.txt"
    log_file.write_text("\n".join(log_lines), encoding="utf-8")
    log(f" -> Log completo salvo: {log_file}")
    log("=" * 78)


if __name__ == "__main__":
    main()