import numpy as np
from .base import BaseChannelCodec
from ..core.structures import BitArray

class ManchesterCodec(BaseChannelCodec):
    """
    Implementação da codificação bifásica Manchester (IEEE 802.3).\
    Convenção: 0 -> [0, 1] (transição baixo->alto), 1 -> [1, 0] (transição alto->baixo).
    """

    def __init__(self, strict_validation: bool = False) -> None:
        self.strict_validation = strict_validation

    def encode(self, bits: BitArray) -> np.ndarray:
        if bits.size == 0: # Retorna um array vazio se o vetor de bits estiver vazio
            return np.array([], dtype=np.uint8)

        # Mapeamento: 0 -> [0, 1], 1 -> [1, 0]
        encoded = np.empty((bits.size, 2), dtype=np.uint8)
        encoded[bits == 0] = [0, 1] # Mapeia 0 para [0, 1]
        encoded[bits == 1] = [1, 0] # Mapeia 1 para [1, 0]
        return encoded.flatten()

    def decode(self, symbols: np.ndarray) -> BitArray:
        if symbols.size == 0: # Retorna um array vazio se o vetor de símbolos estiver vazio
            return np.array([], dtype=np.uint8)

        if symbols.size % 2 != 0: # Verifica se o tamanho do array é múltiplo de 2
            raise ValueError(f"Sequência Manchester inválida: tamanho ({symbols.size}) não é múltiplo de 2.")

        # Reorganiza em pares para validação das transições
        pairs = symbols.reshape(-1, 2)

        if self.strict_validation:
            # Validação das transições Manchester
            zeros = (pairs[:, 0] == 0) & (pairs[:, 1] == 1) # Transição válida para 0
            ones = (pairs[:, 0] == 1) & (pairs[:, 1] == 0) # Transição válida para 1
            invalid = ~(zeros | ones) # Identifica pares inválidos que não correspondem a transições Manchester

            if np.any(invalid):
                # Emulação de violação de código na camada física
                primeiro_erro = int(np.where(invalid)[0][0])
                raise ValueError(f"Violação de código Manchester detetada no símbolo de índice {primeiro_erro}.")

            # Atribui os bits decodificados com base nas transições válidas
            decoded = np.empty(pairs.shape[0], dtype=np.uint8)
            decoded[zeros] = 0
            decoded[ones] = 1
            return decoded

        # Decisão diferencial por meio-bit para simulação de camada física (sem abortar o quadro)
        return (pairs[:, 0] > pairs[:, 1]).astype(np.uint8)