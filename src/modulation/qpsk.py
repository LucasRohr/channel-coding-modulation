import numpy as np
from scipy.special import erfc
from .base import BaseModulator
from ..core.structures import BitArray, IQSignal

class QPSKModulator(BaseModulator):
    """
    Modulador QPSK (Quadrature Phase Shift Keying) com símbolos {+1, -1} em cada eixo I/Q.
    """

    def __init__(self) -> None:
        # Normalização de energia por símbolo unitária (Es = 1)
        self._norm_factor: float = 1.0 / np.sqrt(2.0)

    @property
    def bits_per_symbol(self) -> int:
        # QPSK transmite 2 bits por símbolo, pois cada símbolo representa uma combinação de dois bits (00, 01, 10, 11).
        return 2

    @property
    def spectral_efficiency(self) -> float:
        return 2.0  # 2 bits/s/Hz sob filtragem de Nyquist ideal

    def modulate(self, bits: BitArray) -> IQSignal:
        if bits.size == 0:
            return np.array([], dtype=np.complex128)

        if bits.size % 2 != 0:
            raise ValueError(f"Vetor de entrada QPSK precisa de ter tamanho par; recebido: {bits.size}.")

        # Separação dos canais em fase (I) e quadratura (Q)
        dibits = bits.reshape(-1, 2)
        b_i = dibits[:, 0]
        b_q = dibits[:, 1]

        # Mapeamento: 0 -> +norm, 1 -> -norm
        i_component = np.where(b_i == 0, self._norm_factor, -self._norm_factor)
        q_component = np.where(b_q == 0, self._norm_factor, -self._norm_factor)

        return (i_component + 1j * q_component).astype(np.complex128)

    def demodulate(self, rx_signal: IQSignal) -> BitArray:
        if rx_signal.size == 0:
            return np.array([], dtype=np.uint8)

        num_symbols = rx_signal.size # Número de símbolos QPSK recebidos
        demod_bits = np.empty((num_symbols, 2), dtype=np.uint8) # Array para armazenar os bits demodulados

        # Decisão de mínima distância para cada componente I/Q
        demod_bits[:, 0] = (np.real(rx_signal) < 0.0).astype(np.uint8) # Decisão para o canal I (fase)
        demod_bits[:, 1] = (np.imag(rx_signal) < 0.0).astype(np.uint8) # Decisão para o canal Q (quadratura)

        return demod_bits.flatten()

    # Cálculo da taxa de erro de bit teórica para QPSK em canal AWGN
    def theoretical_ber(self, eb_no_db: np.ndarray) -> np.ndarray:
        eb_no_linear = 10.0 ** (eb_no_db / 10.0)
        # Com codificação Gray, a BER teórica sob Eb/N0 coincide estritamente com BPSK
        return 0.5 * erfc(np.sqrt(eb_no_linear))