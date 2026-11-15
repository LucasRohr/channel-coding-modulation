from abc import ABC, abstractmethod
import numpy as np
from ..core.structures import BitArray, IQSignal

class BaseModulator(ABC):
    """
    Interface para esquemas de modulação digital em banda passante.
    Modela o mapeamento I/Q complexo e a decodificação por mínima distância euclidiana.
    """

    @property
    @abstractmethod
    def bits_per_symbol(self) -> int:
        """Número de bits transmitidos em cada símbolo (k = log2(M))."""
        pass

    @property
    @abstractmethod
    def spectral_efficiency(self) -> float:
        """Eficiência espectral do esquema em bps/Hz."""
        pass

    @abstractmethod
    def modulate(self, bits: BitArray) -> IQSignal:
        """Modula o fluxo binário em coordenadas complexas no espaço de sinais."""
        pass

    @abstractmethod
    def demodulate(self, rx_signal: IQSignal) -> BitArray:
        """Demodula o sinal ruidoso através da decisão de mínima distância."""
        pass

    @abstractmethod
    def theoretical_ber(self, eb_no_db: np.ndarray) -> np.ndarray:
        """Calcula o limite analítico de BER sobre canal AWGN com relação Eb/N0."""
        pass