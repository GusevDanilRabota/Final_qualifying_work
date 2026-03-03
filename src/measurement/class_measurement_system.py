import numpy
import logging
from typing import List
from .class_channel_former import channel_former_t
from .class_probe import probe_t
from .class_quadrature_demodulator import quadrature_demodulator_t

logger = logging.getLogger(__name__)

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