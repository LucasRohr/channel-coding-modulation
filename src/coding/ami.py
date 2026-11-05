import numpy as np
from .base import BaseChannelCodec
from ..core.structures import BitArray

class AMICodec(BaseChannelCodec):
    """
    Codificação Bipolar AMI (Alternate Mark Inversion).
    Converte bits binários num código de linha pseudoternário {-1, 0, +1}.
    """

    def encode(self, bits: BitArray) -> np.ndarray:
        if bits.size == 0: # Retorna um array vazio se o vetor de bits estiver vazio
            return np.array([], dtype=np.int8)

        encoded = np.zeros(bits.size, dtype=np.int8)
        polaridade_atual = 1 # Inicia a polaridade da marca como positiva (+1)

        for idx, bit in enumerate(bits):
            # Se o bit for 1, atribui a polaridade atual e alterna a polaridade para o próximo bit 1
            # Bit 0 é representado como 0 (sem pulso)
            if bit == 1:
                encoded[idx] = polaridade_atual
                polaridade_atual = -polaridade_atual  # Inversão alternada de polaridade

        return encoded

    def decode(self, symbols: np.ndarray) -> BitArray:
        if symbols.size == 0: # Retorna um array vazio se o vetor de símbolos estiver vazio
            return np.array([], dtype=np.uint8)

        # Regra de decisão física: tensão nula -> 0; qualquer pulso (+1 ou -1) -> 1
        decoded = (symbols != 0).astype(np.uint8)
        return decoded