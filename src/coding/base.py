from abc import ABC, abstractmethod
import numpy as np
from ..core.structures import BitArray

class BaseChannelCodec(ABC):
    """
    Interface abstrata para codificadores e decodificadores de canal/linha.
    Garante o encapsulamento das operações de redundância e sinalização.
    """

    @abstractmethod
    def encode(self, bits: BitArray) -> np.ndarray:
        """
        Codifica um vetor binário bruto de entrada.
        
        :param bits: BitArray contendo valores discretos em {0, 1}.
        :return: np.ndarray contendo o fluxo codificado (bits ou níveis lógicos).
        """
        pass

    @abstractmethod
    def decode(self, symbols: np.ndarray) -> BitArray:
        """
        Decodifica os símbolos recebidos recuperando o fluxo binário original.
        
        :param symbols: np.ndarray com os símbolos/níveis recebidos da receção.
        :return: BitArray com os bits originais reconstituídos.
        """
        pass