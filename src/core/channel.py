from abc import ABC, abstractmethod
import numpy as np
from .structures import IQSignal


class BaseChannel(ABC):
    """
    Interface abstrata para modelagem do meio físico de transmissão.
    """

    @abstractmethod
    def transmit(
        self,
        signal: IQSignal,
        eb_no_db: float,
        bits_per_symbol: int = 1,
        code_rate: float = 1.0,
    ) -> IQSignal:
        """
        Propaga o sinal complexo pelo meio físico adicionando distorções

        :param signal: Sinal complexo emitido pelo transmissor
        :param eb_no_db: Relação Eb/N0 desejada em decibéis (dB)
        :param bits_per_symbol: Quantidade de bits transportados por símbolo (k)
        :param code_rate: Taxa efetiva do codificador de canal (k_info / n_codificado)
        :return: Sinal complexo recebido após degradação no canal
        """
        pass


class AWGNChannel(BaseChannel):
    """
    Canal com Ruído Branco Gaussiano Aditivo (Additive White Gaussian Noise)
    Modela o ruído térmico contínuo
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def transmit(
        self,
        signal: IQSignal,
        eb_no_db: float,
        bits_per_symbol: int = 1,
        code_rate: float = 1.0,
    ) -> IQSignal:
        if signal.size == 0: # Se for um sinal vazio, retorna uma cópia vazia
            return signal.copy()

        # Potência média por símbolo transmitido (Es)
        symbol_energy = float(np.mean(np.abs(signal) ** 2))
        if symbol_energy == 0.0: # Se a energia do símbolo for zero, define como 1.0 para evitar divisão por zero
            symbol_energy = 1.0

        # Conversão de Eb/N0 [dB] para escala linear
        eb_no_linear = 10.0 ** (eb_no_db / 10.0)

        # Relação Es/N0 ajustada pela taxa de código e ordem de modulação
        es_no_linear = eb_no_linear * bits_per_symbol * code_rate
        noise_psd_n0 = symbol_energy / es_no_linear

        # Desvio padrão por dimensão ortogonal (I e Q)
        sigma = np.sqrt(noise_psd_n0 / 2.0)

        # Gera ruído gaussiano aditivo para as componentes I e Q
        noise_i = self._rng.normal(0.0, sigma, size=signal.size)
        noise_q = self._rng.normal(0.0, sigma, size=signal.size)

        # Retorna o sinal recebido com ruído aditivo
        return (signal + noise_i + 1j * noise_q).astype(np.complex128)


class ImpulsiveBurstChannel(AWGNChannel):
    """
    Canal AWGN sujeito a ruído impulsivo em rajadas (Burst Errors)
    Simula interferências de curta duração que afetam símbolos consecutivos
    """

    # burst_probability: Probabilidade de ocorrência de uma rajada em cada símbolo transmitido
    # burst_length: Número de símbolos consecutivos afetados por uma rajada
    # impulse_amplitude: Amplitude do ruído impulsivo aditivo
    # seed: Semente para o gerador de números aleatórios (opcional)
    def __init__(
        self,
        burst_probability: float = 0.005,
        burst_length: int = 8,
        impulse_amplitude: float = 5.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        if not 0.0 <= burst_probability <= 1.0:
            raise ValueError("burst_probability deve estar entre 0.0 e 1.0.")
        self.burst_probability = burst_probability
        self.burst_length = burst_length
        self.impulse_amplitude = impulse_amplitude

    def transmit(
        self,
        signal: IQSignal,
        eb_no_db: float,
        bits_per_symbol: int = 1,
        code_rate: float = 1.0,
    ) -> IQSignal:
        # Primeiro aplica o ruído térmico de fundo (AWGN)
        rx_signal = super().transmit(signal, eb_no_db, bits_per_symbol, code_rate)

        # Se o sinal recebido estiver vazio ou a probabilidade de rajada for zero,
        # retorna o sinal recebido sem alterações
        if rx_signal.size == 0 or self.burst_probability == 0.0:
            return rx_signal

        # Sorteia os pontos de início aleatórios de rajadas impulsivas
        burst_starts = self._rng.random(rx_signal.size) < self.burst_probability
        start_indices = np.where(burst_starts)[0]

        for idx in start_indices:
            # Final da rajada não pode ultrapassar o tamanho do sinal
            end_idx = min(idx + self.burst_length, rx_signal.size)
            span = end_idx - idx

            # Rajada de ruído impulsivo aditivo com distribuição normal para as componentes I e Q
            impulse_i = self._rng.normal(0.0, self.impulse_amplitude, size=span)
            impulse_q = self._rng.normal(0.0, self.impulse_amplitude, size=span)

            # Adiciona a rajada de ruído impulsivo ao sinal recebido
            rx_signal[idx:end_idx] += impulse_i + 1j * impulse_q

        # Retorna o sinal recebido com ruído térmico e rajadas impulsivas
        return rx_signal