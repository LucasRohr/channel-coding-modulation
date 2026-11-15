import numpy as np
from src.core.source import DataStream
from src.coding.manchester import ManchesterCodec
from src.coding.ami import AMICodec
from src.coding.hamming import Hamming74Codec
from src.modulation.bpsk import BPSKModulator
from src.modulation.qpsk import QPSKModulator

# Teste de pipeline completo sem ruído
def test_pipeline_zero_noise() -> None:
    message = "Redes Unisinos 2026"
    bits = DataStream.text_to_bits(message)
    
    # 1. Validação Manchester
    m_codec = ManchesterCodec()
    assert np.array_equal(m_codec.decode(m_codec.encode(bits)), bits)

    # 2. Validação AMI
    ami_codec = AMICodec()
    assert np.array_equal(ami_codec.decode(ami_codec.encode(bits)), bits)

    # 3. Validação Hamming(7,4) e Injeção de 1 Erro
    h_codec = Hamming74Codec()
    h_enc = h_codec.encode(bits)
    # Introduzir erro no primeiro bloco (bit 0)
    h_enc_err = h_enc.copy()
    h_enc_err[0] ^= 1
    assert np.array_equal(h_codec.decode(h_enc_err), bits)

    # 4. Validação Modulação BPSK e QPSK (Back-to-Back)
    bpsk = BPSKModulator()
    assert np.array_equal(bpsk.demodulate(bpsk.modulate(bits)), bits)

    qpsk = QPSKModulator()
    assert np.array_equal(qpsk.demodulate(qpsk.modulate(bits)), bits)

    print("Todos os testes da Fase 2 foram validados com 100% de precisão.")

if __name__ == "__main__":
    test_pipeline_zero_noise()