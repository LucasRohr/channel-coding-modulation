import numpy as np
from .base import BaseChannelCodec
from ..core.structures import BitArray

class Hamming74Codec(BaseChannelCodec):
    """
    Código de bloco sistemático Hamming (7,4) para detecção e correção direta de erros.
    Capacidade: Detecção de até 2 erros e correção garantida de 1 erro por bloco.
    """

    def encode(self, bits: BitArray) -> np.ndarray:
        if bits.size == 0: # Retorna um array vazio se o vetor de bits estiver vazio
            return np.array([], dtype=np.uint8)

        # Ajuste de tamanho para múltiplos de 4 (preenchimento com zeros)
        remainder = bits.size % 4
        padded_bits = bits
        self._padding_len = 0

        # Se houver bits restantes, adiciona zeros para completar o bloco de 4 bits
        if remainder != 0:
            self._padding_len = 4 - remainder
            padded_bits = np.concatenate([bits, np.zeros(self._padding_len, dtype=np.uint8)])

        num_blocks = padded_bits.size // 4 # Total de blocos de 4 bits
        encoded = np.empty((num_blocks, 7), dtype=np.uint8) # Array para armazenar os blocos codificados de 7 bits
        data_blocks = padded_bits.reshape(-1, 4) # Reshape para blocos de 4 bits

        # Mapeamento sistemático: d1=pos 3, d2=pos 5, d3=pos 6, d4=pos 7
        d1 = data_blocks[:, 0]
        d2 = data_blocks[:, 1]
        d3 = data_blocks[:, 2]
        d4 = data_blocks[:, 3]

        # Paridades com soma módulo 2 (XOR) para serem inseridos nas posições 1, 2 e 4 do bloco codificado
        p1 = d1 ^ d2 ^ d4
        p2 = d1 ^ d3 ^ d4
        p4 = d2 ^ d3 ^ d4

        # Montagem do bloco codificado de 7 bits: [p1, p2, d1, p4, d2, d3, d4]
        encoded[:, 0] = p1
        encoded[:, 1] = p2
        encoded[:, 2] = d1
        encoded[:, 3] = p4
        encoded[:, 4] = d2
        encoded[:, 5] = d3
        encoded[:, 6] = d4

        return encoded.flatten()

    def decode(self, symbols: np.ndarray) -> BitArray:
        if symbols.size == 0: # Retorna um array vazio se o vetor de símbolos estiver vazio
            return np.array([], dtype=np.uint8)

        if symbols.size % 7 != 0: # Verifica se o tamanho do vetor de símbolos é múltiplo de 7, caso contrário, lança um erro
            raise ValueError(f"Comprimento do vetor Hamming inválido: {symbols.size} não é múltiplo de 7.")

        blocks = symbols.reshape(-1, 7).copy() # Cria uma cópia dos blocos de 7 bits para evitar alterações no array original
        num_blocks = blocks.shape[0]

        # Cálculo do vetor de síndrome: S = [s4 s2 s1]
        s1 = blocks[:, 0] ^ blocks[:, 2] ^ blocks[:, 4] ^ blocks[:, 6]
        s2 = blocks[:, 1] ^ blocks[:, 2] ^ blocks[:, 5] ^ blocks[:, 6]
        s4 = blocks[:, 3] ^ blocks[:, 4] ^ blocks[:, 5] ^ blocks[:, 6]

        syndromes = s1 * 1 + s2 * 2 + s4 * 4

        # Correção de erro de 1 bit
        for i in range(num_blocks):
            error_pos = syndromes[i]
            if error_pos > 0:
                # O índice do erro em base-0 é error_pos - 1
                # Corrige o bit com erro invertendo seu valor (XOR com 1)
                blocks[i, error_pos - 1] ^= 1

        # Extração dos bits de dados úteis (posições 2, 4, 5, 6)
        recovered_data = blocks[:, [2, 4, 5, 6]].flatten()

        # Remoção do padding se inserido durante a codificação
        if hasattr(self, '_padding_len') and self._padding_len > 0:
            recovered_data = recovered_data[:-self._padding_len]

        return recovered_data