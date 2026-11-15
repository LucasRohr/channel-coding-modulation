import numpy as np
from typing import Optional
from .structures import BitArray, IQSignal
from ..coding.base import BaseChannelCodec
from ..coding.ami import AMICodec
from ..modulation.base import BaseModulator


class Receiver:
    """
    Representa a entidade receptora do sistema de comunicação digital
    Executa a demodulação/amostragem e a decodificação de canal
    """

    def __init__(
        self,
        codec: Optional[BaseChannelCodec] = None,
        modulator: Optional[BaseModulator] = None,
    ) -> None:
        self.codec = codec
        self.modulator = modulator

    def receive(self, rx_signal: IQSignal, mod_padding_bits: int = 0) -> BitArray:
        """
        Executa o pipeline completo de recepção.

        :param rx_signal: Sinal complexo ruidoso proveniente do canal
        :param mod_padding_bits: Bits de preenchimento a descartar após demodulação
        :return: BitArray recuperado para entrega à camada superior
        """

        # Se o sinal recebido estiver vazio, retorna um fluxo de bits vazio
        if rx_signal.size == 0:
            return np.array([], dtype=np.uint8)

        # 1. Demodulação ou Decisão de Nível em Banda Base

        # Se houver modulador definido, realiza a demodulação do sinal recebido
        if self.modulator is not None:
            # Se houver bits de preenchimento, descarta-os após a demodulação

            demodulated = self.modulator.demodulate(rx_signal)
            if mod_padding_bits > 0:
                demodulated = demodulated[:-mod_padding_bits]
        else:
            # Caso não haja modulador, realiza a decisão de nível diretamente no sinal recebido

            real_part = np.real(rx_signal)
            if isinstance(self.codec, AMICodec):
                # Caso seja AMI, aplica limiares de decisão ternária {-1, 0, +1} em -0.5 V e +0.5 V

                demodulated = np.where(
                    real_part > 0.5, 1, np.where(real_part < -0.5, -1, 0)
                ).astype(np.int8)
            else:
                # Caso contrário, aplica limiar NRZ Polar em 0.0 V, mapeando valores positivos para 0 e negativos para 1

                # Limiar NRZ Polar em 0.0 V
                demodulated = (real_part < 0.0).astype(np.uint8)

        # 2. Decodificação de Canal / Linha
        if self.codec is not None:
            recovered_bits = self.codec.decode(demodulated)
        else:
            recovered_bits = demodulated.astype(np.uint8)

        return recovered_bits