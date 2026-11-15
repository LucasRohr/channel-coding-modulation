import numpy as np
from typing import Optional, Tuple
from .structures import BitArray, IQSignal
from ..coding.base import BaseChannelCodec
from ..coding.ami import AMICodec
from ..modulation.base import BaseModulator


class Transmitter:
    """
    Representa a entidade transmissora do sistema de comunicação digital.
    Coordena a codificação de canal e a modulação para envio ao meio físico.
    """

    def __init__(
        self,
        codec: Optional[BaseChannelCodec] = None,
        modulator: Optional[BaseModulator] = None,
    ) -> None:
        self.codec = codec # Codificador de canal (opcional)
        self.modulator = modulator # Modulador digital (opcional)
        self._last_code_rate: float = 1.0 # Taxa de código efetiva do último fluxo codificado
        self._mod_padding_bits: int = 0 # Quantidade de bits de preenchimento adicionados para fechar símbolos

    @property
    def effective_code_rate(self) -> float:
        """Retorna a taxa de código efetiva (bits de informação / símbolos codificados)."""
        return self._last_code_rate

    @property
    def modulation_padding(self) -> int:
        """Retorna a quantidade de bits de preenchimento adicionados para fechar símbolos."""
        return self._mod_padding_bits

    def transmit(self, raw_bits: BitArray) -> Tuple[IQSignal, np.ndarray]:
        """
        Executa o pipeline completo de transmissão.

        :param raw_bits: Fluxo binário original vindo da fonte de dados.
        :return: Tupla (sinal_complexo_tx, fluxo_codificado).
        """

        # Se o fluxo de bits de entrada estiver vazio, retorna sinais vazios
        if raw_bits.size == 0:
            return np.array([], dtype=np.complex128), np.array([], dtype=np.uint8)

        # 1. Codificação de Canal / Linha
        if self.codec is not None:
            encoded_stream = self.codec.encode(raw_bits)

            # Atualiza a taxa de código efetiva com base no tamanho do fluxo de bits original e codificado
            self._last_code_rate = raw_bits.size / encoded_stream.size
        else:
            # Se não houver codificação, o fluxo codificado é uma cópia do fluxo original e a taxa de código é 1.0
            encoded_stream = raw_bits.copy()
            self._last_code_rate = 1.0

        # 2. Modulação Digital (Banda Passante) ou Sinalização em Banda Base
        if self.modulator is not None:
            if isinstance(self.codec, AMICodec):
                raise TypeError(
                    "AMICodec gera símbolos pseudoternários {-1, 0, +1} de banda base "
                    "e não pode ser acoplado diretamente a moduladores binários."
                )

            k = self.modulator.bits_per_symbol # Número de bits transportados por símbolo
            remainder = encoded_stream.size % k # Resto da divisão do tamanho do fluxo codificado pelo número de bits por símbolo

            # Quantidade de bits de preenchimento necessários para fechar o último símbolo
            self._mod_padding_bits = (k - remainder) if remainder != 0 else 0

            # Se houver bits de preenchimento, adiciona zeros ao final do fluxo codificado
            if self._mod_padding_bits > 0:
                padded_stream = np.concatenate(
                    [encoded_stream, np.zeros(self._mod_padding_bits, dtype=np.uint8)]
                )
            else:
                # Caso não haja bits de preenchimento, o fluxo codificado permanece inalterado
                padded_stream = encoded_stream

            tx_signal = self.modulator.modulate(padded_stream) # Realiza modulação digital do fluxo codificado 
        else:
            # Transmissão direta em Banda Base caso não haja modulador definido
            self._mod_padding_bits = 0
            if isinstance(self.codec, AMICodec):
                # Níveis pseudoternários {-1.0, 0.0, +1.0}
                tx_signal = encoded_stream.astype(np.complex128)
            else:
                # Mapeamento NRZ Polar: 0 -> +1.0 V, 1 -> -1.0 V[cite: 257]
                tx_signal = np.where(encoded_stream == 0, 1.0, -1.0).astype(np.complex128)

        # Retorna o sinal complexo transmitido e o fluxo codificado (com ou sem bits de preenchimento)
        return tx_signal, encoded_stream