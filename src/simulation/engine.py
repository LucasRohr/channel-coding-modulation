from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..core.channel import BaseChannel
from ..core.receiver import Receiver
from ..core.source import DataStream
from ..core.structures import BitArray, IQSignal, SimulationMetrics
from ..core.transmitter import Transmitter
from ..coding.base import BaseChannelCodec
from ..modulation.base import BaseModulator

# Arquivo do módulo responsável por métricas de desempenho e simulação Monte Carlo para avaliação de BER x SNR e demonstrações ASCII.

@dataclass
class ScenarioConfig:
    """Define a configuração completa de um enlace digital simulado."""
    name: str
    codec: Optional[BaseChannelCodec]
    modulator: Optional[BaseModulator]
    channel: BaseChannel


@dataclass
class AsciiTransmissionResult:
    """Armazena o resultado detalhado da transmissão de uma mensagem ASCII."""
    scenario_name: str
    eb_no_db: float # Razão energia por bit/ruído em dB usada na simulação
    original_text: str
    recovered_text: str
    total_bits: int # Número total de bits transmitidos
    bit_errors: int # Número de bits incorretos na recepção
    ber: float # Taxa de erro de bits (BER) = bit_errors / total_bits
    transmitter_signal_sample: IQSignal # Amostra do sinal transmitido
    receiver_signal_sample: IQSignal # Amostra do sinal recebido


class MonteCarloSimulator:
    """
    Orquestrador estatístico para avaliação de BER x SNR e demonstrações ASCII.
    """

    def __init__(self, seed: int = 2026) -> None:
        self._rng = np.random.default_rng(seed)

    def simulate_ascii_message(
        self,
        message: str,
        scenario: ScenarioConfig,
        eb_no_db: float,
    ) -> AsciiTransmissionResult:
        """
        Executa o ciclo completo de comunicação para uma mensagem ASCII real:
        ASCII -> Bits -> Codificação -> Modulação -> Canal -> Demodulação -> Decodificação -> ASCII.
        """
        raw_bits = DataStream.text_to_bits(message)
        transmitter = Transmitter(codec=scenario.codec, modulator=scenario.modulator)
        receiver = Receiver(codec=scenario.codec, modulator=scenario.modulator)

        # Transmissão do sinal pelo transmissor e pelo canal
        transmitter_signal, _ = transmitter.transmit(raw_bits)
        bits_per_sym = scenario.modulator.bits_per_symbol if scenario.modulator else 1

        # Recebimento do sinal pelo canal com ruído adicionado
        receiver_signal = scenario.channel.transmit(
            signal=transmitter_signal,
            eb_no_db=eb_no_db,
            bits_per_symbol=bits_per_sym,
            code_rate=transmitter.effective_code_rate,
        )

        try:
            recovered_bits = receiver.receive(receiver_signal, mod_padding_bits=transmitter.modulation_padding)
            bit_errors = int(np.sum(raw_bits != recovered_bits))
            recovered_text = DataStream.bits_to_text(recovered_bits)
        except ValueError as exc:
            # Violações estruturais (ex.: violação de transição Manchester) contam como erro de quadro
            bit_errors = raw_bits.size
            recovered_text = f"[ERRO DE QUADRO: {exc}]"

        # Calcula a taxa de erro de bits (BER)
        ber = bit_errors / raw_bits.size if raw_bits.size > 0 else 0.0

        return AsciiTransmissionResult(
            scenario_name=scenario.name,
            eb_no_db=eb_no_db,
            original_text=message,
            recovered_text=recovered_text,
            total_bits=int(raw_bits.size),
            bit_errors=bit_errors,
            ber=ber,
            transmitter_signal_sample=transmitter_signal,
            receiver_signal_sample=receiver_signal,
        )

    def run_ber_curve(
        self,
        scenario: ScenarioConfig,
        eb_no_db_range: np.ndarray,
        batch_size: int = 50_000,
        min_errors: int = 150,
        max_bits: int = 1_000_000,
    ) -> List[SimulationMetrics]:
        """
        Executa simulação Monte Carlo adaptativa para levantar a curva BER x Eb/N0.
        """
        transmitter = Transmitter(codec=scenario.codec, modulator=scenario.modulator)
        receiver = Receiver(codec=scenario.codec, modulator=scenario.modulator)
        bits_per_sym = scenario.modulator.bits_per_symbol if scenario.modulator else 1

        results: List[SimulationMetrics] = []

        # Executa a simulação para cada valor de Eb/N0 especificado
        for eb_no_db in eb_no_db_range:
            total_bits = 0
            total_errors = 0

            # Executa a simulação em lotes até atingir o número mínimo de erros ou o número máximo de bits
            while total_bits < max_bits and total_errors < min_errors:
                bits: BitArray = self._rng.integers(0, 2, size=batch_size, dtype=np.uint8)

                # Transmissão do sinal pelo transmissor e pelo canal
                transmitter_signal, _ = transmitter.transmit(bits)

                receiver_signal = scenario.channel.transmit(
                    signal=transmitter_signal,
                    eb_no_db=float(eb_no_db),
                    bits_per_symbol=bits_per_sym,
                    code_rate=transmitter.effective_code_rate,
                )

                try:
                    # Recebe os bits do sinal recebido e calcula o número de erros
                    rec_bits = receiver.receive(receiver_signal, mod_padding_bits=transmitter.modulation_padding)
                    errors = int(np.sum(bits != rec_bits))
                except ValueError:
                    errors = batch_size

                # Atualiza o total de bits e erros acumulados
                total_bits += batch_size
                total_errors += errors

            # Calcula a taxa de erro de bits (BER) teórica se o modulador estiver definido e o codec não estiver definido
            theo_ber: Optional[float] = None
            if scenario.modulator is not None and scenario.codec is None:
                theo_ber = float(scenario.modulator.theoretical_ber(np.array([eb_no_db]))[0])

            # Calcula a eficiência espectral efetiva do esquema de modulação/codificação
            eff_spec = bits_per_sym * transmitter.effective_code_rate

            # Adiciona os resultados da simulação para o valor atual de Eb/N0
            results.append(
                SimulationMetrics(
                    total_bits=total_bits,
                    error_bits=total_errors,
                    eb_no_db=float(eb_no_db),
                    ber_theoretical=theo_ber,
                    spectral_efficiency=eff_spec,
                )
            )

        return results

    def capture_constellation_snapshot(
        self,
        modulator: BaseModulator,
        channel: BaseChannel,
        eb_no_db: float,
        num_bits: int = 4_000,
    ) -> Tuple[IQSignal, IQSignal]:
        """Gera amostras I/Q transmitidas e recebidas para plotagem de constelação."""

        # Gera bits aleatórios para modulação
        bits = self._rng.integers(0, 2, size=num_bits, dtype=np.uint8)

        transmitter_signal = modulator.modulate(bits)
        receiver_signal = channel.transmit(
            signal=transmitter_signal,
            eb_no_db=eb_no_db,
            bits_per_symbol=modulator.bits_per_symbol,
            code_rate=1.0,
        )

        return transmitter_signal, receiver_signal