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
    tx = Transmitter(codec=Hamming74Codec(), modulator=QPSKModulator())
    rx = Receiver(codec=Hamming74Codec(), modulator=QPSKModulator())
    channel = AWGNChannel(seed=42) # Seed 42 para reprodutibilidade do teste

    tx_sig, _ = tx.transmit(raw_bits)
    rx_sig = channel.transmit(
        tx_sig,
        eb_no_db=12.0,
        bits_per_symbol=QPSKModulator().bits_per_symbol,
        code_rate=tx.effective_code_rate,
    )
    recovered_bits = rx.receive(rx_sig, mod_padding_bits=tx.modulation_padding)
    recovered_msg = DataStream.bits_to_text(recovered_bits)

    assert recovered_msg == msg, f"Esperado '{msg}', obtido '{recovered_msg}'"

    # 2. Cenário Banda Base: AMI Bipolar em SNR alta (14 dB)
    tx_ami = Transmitter(codec=AMICodec(), modulator=None)
    rx_ami = Receiver(codec=AMICodec(), modulator=None)

    tx_sig_ami, _ = tx_ami.transmit(raw_bits)
    rx_sig_ami = channel.transmit(
        tx_sig_ami,
        eb_no_db=14.0,
        bits_per_symbol=1,
        code_rate=tx_ami.effective_code_rate,
    )
    rec_bits_ami = rx_ami.receive(rx_sig_ami)
    assert DataStream.bits_to_text(rec_bits_ami) == msg

    # 3. Verificação estatística rápida de BER (BPSK não codificado a 4 dB)
    rng = np.random.default_rng(123)
    monte_carlo_bits = rng.integers(0, 2, size=200_000, dtype=np.uint8)
    bpsk = BPSKModulator()
    tx_mc = Transmitter(codec=None, modulator=bpsk)
    rx_mc = Receiver(codec=None, modulator=bpsk)

    sig_mc, _ = tx_mc.transmit(monte_carlo_bits)
    rx_mc_sig = channel.transmit(sig_mc, eb_no_db=4.0, bits_per_symbol=1, code_rate=1.0)
    rec_mc = rx_mc.receive(rx_mc_sig)

    ber_sim = np.mean(monte_carlo_bits != rec_mc)
    ber_teo = float(bpsk.theoretical_ber(np.array([4.0]))[0])
    assert abs(ber_sim - ber_teo) < 0.002, f"Desvio excessivo: sim={ber_sim:.5f}, teo={ber_teo:.5f}"

    print(f"Feito. Mensagem recuperada: '{recovered_msg}'")
    print(f"Calibracao AWGN (4 dB BPSK) -> BER Simulada: {ber_sim:.5f} | BER Teorica: {ber_teo:.5f}")


if __name__ == "__main__":
    test_end_to_end_transmission()