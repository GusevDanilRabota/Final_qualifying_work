# measurement.py
import numpy
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class probe_t:
    """
    Измерительная головка с четырьмя электродами.
    Электроды расположены симметрично относительно центра головки.
    """
    def __init__(self, a: float) -> None:
        """
        a – размер головки (расстояние между крайними электродами), м.
        """
        if a <= 0:
            raise ValueError("Размер головки должен быть положительным")
        self.a: float = a

    def get_electrode_coords(self, xc: float) -> List[float]:
        """
        Возвращает координаты четырёх электродов для центра головки xc.
        xc – координата центра головки вдоль линии (м).
        """
        return [xc + self.a/2, xc + self.a/2, xc - self.a/2, xc - self.a/2]


class channel_former_t:
    """
    Формирует суммарный и разностные каналы из напряжений на электродах.
    """
    @staticmethod
    def form_channels(V: List[complex]) -> Tuple[complex, complex, complex]:
        """
        Принимает список напряжений на четырёх электродах V = [v1, v2, v3, v4].
        Возвращает кортеж (S, Dx, Dy):
          S  = v1 + v2 + v3 + v4                     (суммарный канал)
          Dx = (v1 + v2) - (v3 + v4)                 (разностный по x)
          Dy = (v1 + v4) - (v3 + v2)                 (разностный по y)
        """
        if len(V) != 4:
            raise ValueError("Должно быть ровно 4 напряжения")
        v1, v2, v3, v4 = V
        S = v1 + v2 + v3 + v4
        Dx = (v1 + v2) - (v3 + v4)
        Dy = (v1 + v4) - (v3 + v2)
        return S, Dx, Dy


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


class measurement_system_t:
    """
    Измерительная система, которая перемещает головку вдоль линии,
    на каждой частоте измеряет напряжения и формирует признаки.
    """
    def __init__(
        self,
        probe: probe_t,
        channel_former: channel_former_t,
        demodulator: quadrature_demodulator_t,
        frequencies: numpy.ndarray,
        P0: float = 1.0
    ) -> None:
        """
        probe          – объект probe_t
        channel_former – объект channel_former_t
        demodulator    – объект quadrature_demodulator_t
        frequencies    – массив частот (Гц), на которых производятся измерения
        P0             – мощность падающей волны (Вт)
        """
        self.probe: probe_t = probe
        self.channel_former: channel_former_t = channel_former
        self.demodulator: quadrature_demodulator_t = demodulator
        self.frequencies: numpy.ndarray = numpy.asarray(frequencies)
        if P0 <= 0:
            raise ValueError("Мощность должна быть положительной")
        self.P0: float = P0

    def _compute_U_inc(self, line) -> float:
        """
        Вычисляет амплитуду падающей волны по мощности P0 и волновому
        сопротивлению линии на текущей частоте.
        U_inc = sqrt(2 * P0 * Z0)
        """
        Z0 = line.Z0
        if Z0 is None:
            raise RuntimeError("Волновое сопротивление линии не вычислено. Установите частоту.")
        return numpy.sqrt(2 * self.P0 * Z0)

    def measure(self, line, defect, xc: float) -> numpy.ndarray:
        """
        Выполняет измерение в позиции xc (центр головки) при заданной линии
        и дефекте (может быть None, если дефекта нет).

        Возвращает одномерный массив признаков, сформированный следующим образом:
        для каждой частоты (в порядке self.frequencies) вычисляются комплексные
        значения S, Dx, Dy, затем каждое демодулируется в I и Q, и все эти
        величины конкатенируются в порядке:
        [I_S(f1), Q_S(f1), I_Dx(f1), Q_Dx(f1), I_Dy(f1), Q_Dy(f1),
         I_S(f2), Q_S(f2), ...]

        Предполагается, что внешний код уже гарантирует, что головка не
        перекрывает границу дефекта (чтобы избежать разрывных скачков напряжения).
        """
        features: List[float] = []
        for f in self.frequencies:
            # Устанавливаем частоту линии и дефекта
            line.set_frequency(f)
            if defect is not None:
                defect.set_frequency(f)

            # Координаты электродов
            x_coords = self.probe.get_electrode_coords(xc)

            # Падающая волна
            U_inc = self._compute_U_inc(line)

            # Сбор напряжений на электродах
            V: List[complex] = []
            for x in x_coords:
                if defect is None:
                    # Линия без дефекта – чисто бегущая волна
                    beta = line.beta
                    if beta is None:
                        raise RuntimeError("Постоянная распространения не вычислена")
                    v = U_inc * numpy.exp(-1j * beta * x)
                else:
                    # Линия с дефектом – используем модель дефекта
                    v = defect.voltage_at(x, U_inc)
                V.append(v)

            # Формирование каналов
            S, Dx, Dy = self.channel_former.form_channels(V)

            # Демодуляция и добавление признаков
            for comp in (S, Dx, Dy):
                I, Q = self.demodulator.demodulate(comp)
                features.extend([I, Q])

            logger.debug(f"Частота {f/1e9:.2f} ГГц: признаки добавлены")

        return numpy.array(features)