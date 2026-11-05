import numpy as np
from dataclasses import dataclass
from typing import TypeAlias, Optional

# Aliases matemáticos para o pipeline de transmissão
BitArray: TypeAlias = np.ndarray  # Array unidimensional de np.uint8 contendo apenas 0s e 1s
IQSignal: TypeAlias = np.ndarray  # Array unidimensional de np.complex128 (símbolos no plano I/Q)

"""
Estrutura para armazenamento e cálculo de métricas de desempenho.
Garante que a avaliação do canal AWGN seja feita corretamente.
"""
@dataclass
class SimulationMetrics:
    total_bits: int # Total de bits transmitidos
    error_bits: int # Total de bits recebidos incorretamente
    eb_no_db: float # Relação sinal-ruído por bit (Eb/N0) em decibéis
    
    @property
    def ber_simulated(self) -> float:
        """Calcula a Taxa de Erro de Bit (BER) empírica."""
        if self.total_bits == 0:
            return 0.0
        return self.error_bits / self.total_bits

    ber_theoretical: Optional[float] = None # Taxa de Erro de Bit (BER) teórica
    spectral_efficiency: Optional[float] = None # Eficiência espectral