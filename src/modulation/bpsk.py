import numpy as np
from scipy.special import erfc
from .base import BaseModulator
from ..core.structures import BitArray, IQSignal

class BPSKModulator(BaseModulator):
    """
    Modulador BPSK (Binary Phase Shift Keying) com símbolos {+1, -1}.
    """

    @property
    def bits_per_symbol(self) -> int:
        # BPSK transmite 1 bit por símbolo, pois cada símbolo representa um único bit (0 ou 1).
        return 1

    @property
    def spectral_efficiency(self) -> float:
        return 1.0  # 1 bit/s/Hz sob filtragem de Nyquist ideal

    def modulate(self, bits: BitArray) -> IQSignal:
        if bits.size == 0: # Retorna um array vazio se o vetor de bits estiver vazio
            return np.array([], dtype=np.complex128)

        # Mapeamento: 0 -> +1.0, 1 -> -1.0
        symbols = np.where(bits == 0, 1.0, -1.0)
        return symbols.astype(np.complex128)

    def demodulate(self, rx_signal: IQSignal) -> BitArray:
        if rx_signal.size == 0: # Retorna um array vazio se o vetor de sinais recebidos estiver vazio
            return np.array([], dtype=np.uint8)

        # Decisão no limiar do eixo real (distância euclidiana em relação a 0)
        return (np.real(rx_signal) < 0.0).astype(np.uint8)

    # Cálculo da taxa de erro de bit teórica para BPSK em canal AWGN
    def theoretical_ber(self, eb_no_db: np.ndarray) -> np.ndarray:
        eb_no_linear = 10.0 ** (eb_no_db / 10.0) # Converte Eb/N0 de dB para escala linear
        return 0.5 * erfc(np.sqrt(eb_no_linear)) # Calcula a BER teórica usando a função complementar do erro (erfc)