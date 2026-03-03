import numpy
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class quadrature_demodulator_t:
    """
    Квадратурный демодулятор, выделяющий синфазную (I) и квадратурную (Q)
    составляющие комплексного сигнала. Может добавлять шум для имитации
    реального тракта.
    """
    def __init__(self, snr_db: Optional[float] = None) -> None:
        """
        snr_db – отношение сигнал/шум в дБ. Если None, шум не добавляется.
        """
        self.snr_db: Optional[float] = snr_db

    def demodulate(self, complex_signal: complex) -> Tuple[float, float]:
        """
        Принимает комплексный сигнал, возвращает (I, Q) – вещественные значения.
        Если задан SNR, добавляет гауссов шум к комплексному сигналу.
        """
        if self.snr_db is None:
            return complex_signal.real, complex_signal.imag
        else:
            # Мощность сигнала (комплексная амплитуда)
            signal_power = numpy.abs(complex_signal) ** 2
            # Мощность шума (в единицах сигнала)
            noise_power = signal_power / (10 ** (self.snr_db / 10))
            noise_std = numpy.sqrt(noise_power / 2)
            noise = noise_std * (numpy.random.randn() + 1j * numpy.random.randn())
            noisy = complex_signal + noise
            logger.debug(f"Добавлен шум: SNR={self.snr_db} дБ")
            return noisy.real, noisy.imag