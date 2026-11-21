from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SpectralEfficiencyReport:
    """
    Consolida os parâmetros teóricos e efetivos de capacidade e banda de um esquema
    """
    scheme_name: str # Nome do esquema de modulação/codificação
    bits_per_symbol: int # Número de bits transmitidos por símbolo (modulação)
    code_rate: float # Taxa de código (codificação de canal)
    nominal_efficiency_bps_hz: float # Eficiência espectral nominal (bits/s/Hz) = bits_per_symbol
    effective_efficiency_bps_hz: float # Eficiência espectral efetiva (bits/s/Hz) = bits_per_symbol * code_rate
    net_bitrate_at_1mhz_mbps: float # Taxa de bits líquida em Mbps para largura de banda de 1 MHz
    shannon_min_eb_no_db: float # Limite mínimo de Eb/N0 exigido por Shannon para transmissão livre de erros nessa eficiência


class PerformanceAnalyzer:
    """
    Calcula métricas de eficiência espectral e limites fundamentais de canal
    baseados nos teoremas de Nyquist e Shannon
    """

    @staticmethod
    def compute_spectral_report(
        scheme_name: str,
        bits_per_symbol: int,
        code_rate: float,
        bandwidth_hz: float = 1_000_000.0,
    ) -> SpectralEfficiencyReport:
        """
        Calcula a eficiência espectral nominal e efetiva, além do limite mínimo
        de Eb/N0 exigido por Shannon para transmissão livre de erros nessa eficiência.
        """
        nominal_efficiency = float(bits_per_symbol)
        effective_efficiency = nominal_efficiency * code_rate
        net_bitrate_mbps = (effective_efficiency * bandwidth_hz) / 1e6

        # Limite de Shannon: C/B = log2(1 + (Eb/N0) * (C/B)) => Eb/N0 = (2^eta - 1) / eta
        shannon_eb_no_linear = ((2.0 ** effective_efficiency) - 1.0) / effective_efficiency
        shannon_eb_no_db = float(10.0 * np.log10(shannon_eb_no_linear))

        return SpectralEfficiencyReport(
            scheme_name=scheme_name,
            bits_per_symbol=bits_per_symbol,
            code_rate=code_rate,
            nominal_efficiency_bps_hz=nominal_efficiency,
            effective_efficiency_bps_hz=effective_efficiency,
            net_bitrate_at_1mhz_mbps=net_bitrate_mbps,
            shannon_min_eb_no_db=shannon_eb_no_db,
        )

    @staticmethod
    def eb_no_to_channel_snr_db(eb_no_db: float, effective_efficiency: float) -> float:
        """Converte Eb/N0 [dB] para SNR física do canal S/N [dB]."""
        return float(eb_no_db + 10.0 * np.log10(effective_efficiency))