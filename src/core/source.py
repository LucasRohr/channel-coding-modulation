import numpy as np
from structures import BitArray

"""
Simula a camada de aplicação/apresentação entregando o fluxo de dados bruto.
Responsável pela conversão bidirecional entre texto e fluxo binário.
"""
class DataStream: 
    @staticmethod
    def text_to_bits(text: str) -> BitArray:
        """
        Converte uma string em um BitArray de 8 bits por caractere.
        """
        if not text:
            return np.array([], dtype=np.uint8) # Retorna um array vazio se a string estiver vazia
            
        bits = []
        for char in text:
            byte_val = ord(char)
            if byte_val > 255: # Verifica se o caractere está fora do padrão ASCII estendido
                raise ValueError(f"Caracteres fora do padrão ASCII estendido não são suportados: '{char}'")
            
            # Extrai os bits do caractere em ordem MSB (Most Significant Bit) para LSB (Least Significant Bit)
            char_bits = [int(b) for b in format(byte_val, '08b')]
            bits.extend(char_bits)
            
        return np.array(bits, dtype=np.uint8) # Retorna o array de bits como np.uint8 (0s e 1s)

    @staticmethod
    def bits_to_text(bits: BitArray) -> str:
        """
        Remonta o texto a partir do fluxo binário recebido após decodificação.
        """
        if bits.size == 0:
            return ""
            
        if bits.size % 8 != 0: # Verifica se o tamanho do array é múltiplo de 8
            raise ValueError(f"Falha de integridade do quadro: tamanho do array ({bits.size}) não é múltiplo de 8.")

        chars = []
        # Processamento em blocos de 8 bits
        for i in range(0, bits.size, 8):
            byte_array = bits[i:i+8]
            # Reconstrói o valor numérico (0 a 255)
            byte_val = int("".join(str(b) for b in byte_array), 2)
            chars.append(chr(byte_val))
            
        return "".join(chars) # Retorna a string reconstruída a partir dos caracteres