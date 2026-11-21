import numpy as np
from src.core.source import DataStream
from src.core.transmitter import Transmitter
from src.core.receiver import Receiver
from src.core.channel import AWGNChannel, ImpulsiveBurstChannel
from src.coding.hamming import Hamming74Codec
from src.coding.ami import AMICodec
from src.modulation.bpsk import BPSKModulator
from src.modulation.qpsk import QPSKModulator


def test_end_to_end_transmission() -> None:
    msg = "Camada Fisica e Enlace - UNISINOS 2026"
    raw_bits = DataStream.text_to_bits(msg)

    # 1. Cenário Banda Passante: Hamming(7,4) + QPSK em SNR alta (12 dB)
    transmitter = Transmitter(codec=Hamming74Codec(), modulator=QPSKModulator())
    receiver = Receiver(codec=Hamming74Codec(), modulator=QPSKModulator())
    channel = AWGNChannel(seed=42) # Seed 42 para reprodutibilidade do teste

    transmitter_signal, _ = transmitter.transmit(raw_bits)
    receiver_signal = channel.transmit(
        transmitter_signal,
        eb_no_db=12.0,
        bits_per_symbol=QPSKModulator().bits_per_symbol,
        code_rate=transmitter.effective_code_rate,
    )
    recovered_bits = receiver.receive(receiver_signal, mod_padding_bits=transmitter.modulation_padding)
    recovered_msg = DataStream.bits_to_text(recovered_bits)

    assert recovered_msg == msg, f"Esperado '{msg}', obtido '{recovered_msg}'"

    # 2. Cenário Banda Base: AMI Bipolar em SNR alta (14 dB)
    transmitter_ami = Transmitter(codec=AMICodec(), modulator=None)
    receiver_ami = Receiver(codec=AMICodec(), modulator=None)

    transmitter_signal_ami, _ = transmitter_ami.transmit(raw_bits)
    receiver_signal_ami = channel.transmit(
        transmitter_signal_ami,
        eb_no_db=14.0,
        bits_per_symbol=1,
        code_rate=transmitter_ami.effective_code_rate,
    )
    recovered_bits_ami = receiver_ami.receive(receiver_signal_ami)
    assert DataStream.bits_to_text(recovered_bits_ami) == msg

    # 3. Verificação estatística rápida de BER (BPSK não codificado a 4 dB)
    rng = np.random.default_rng(123)
    monte_carlo_bits = rng.integers(0, 2, size=200_000, dtype=np.uint8)
    bpsk = BPSKModulator()
    transmitter_mc = Transmitter(codec=None, modulator=bpsk)
    receiver_mc = Receiver(codec=None, modulator=bpsk)

    transmitter_signal_mc, _ = transmitter_mc.transmit(monte_carlo_bits)
    receiver_signal_mc = channel.transmit(transmitter_signal_mc, eb_no_db=4.0, bits_per_symbol=1, code_rate=1.0)
    recovered_bits_mc = receiver_mc.receive(receiver_signal_mc)

    ber_sim = np.mean(monte_carlo_bits != recovered_bits_mc)
    ber_teo = float(bpsk.theoretical_ber(np.array([4.0]))[0])
    assert abs(ber_sim - ber_teo) < 0.002, f"Desvio excessivo: sim={ber_sim:.5f}, teo={ber_teo:.5f}"

    print(f"Feito. Mensagem recuperada: '{recovered_msg}'")
    print(f"Calibracao AWGN (4 dB BPSK) -> BER Simulada: {ber_sim:.5f} | BER Teorica: {ber_teo:.5f}")


if __name__ == "__main__":
    test_end_to_end_transmission()