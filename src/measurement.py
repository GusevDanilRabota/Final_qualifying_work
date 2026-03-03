import numpy

class probe_t:
    """
    Измерительная головка с четырьмя электродами.
    Электроды расположены симметрично относительно центра головки.
    """
    def __init__(self, a):
        """
        a – размер головки (расстояние между крайними электродами), м.
        Электроды находятся в точках: центр ± a/2 (два электрода),
        но в данной модели используются четыре электрода, расположенных
        на линии: два слева и два справа от центра? 
        В исходном коде get_electrode_coords возвращает список:
        [xc + a/2, xc + a/2, xc - a/2, xc - a/2] – это два электрода
        справа (одинаковые координаты) и два слева (одинаковые).
        Это упрощение: фактически два электрода с каждой стороны.
        """
        self.a = a

    def get_electrode_coords(self, xc):
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
    def form_channels(V):
        """
        Принимает список напряжений на четырёх электродах V = [v1, v2, v3, v4].
        Возвращает кортеж (S, Dx, Dy):
          S  = v1 + v2 + v3 + v4                     (суммарный канал)
          Dx = (v1 + v2) - (v3 + v4)                 (разностный по x)
          Dy = (v1 + v4) - (v3 + v2)                 (разностный по y)
        """
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
    def __init__(self, snr_db=None):
        """
        snr_db – отношение сигнал/шум в дБ. Если None, шум не добавляется.
        """
        self.snr_db = snr_db

    def demodulate(self, complex_signal):
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
            return noisy.real, noisy.imag


class measurement_system_t:
    """
    Измерительная система, которая перемещает головку вдоль линии,
    на каждой частоте измеряет напряжения и формирует признаки.
    """
    def __init__(self, probe, channel_former, demodulator, frequencies, P0=1.0):
        """
        probe          – объект Probe
        channel_former – объект ChannelFormer
        demodulator    – объект QuadratureDemodulator
        frequencies    – массив частот (Гц), на которых производятся измерения
        P0             – мощность падающей волны (Вт)
        """
        self.probe = probe
        self.channel_former = channel_former
        self.demodulator = demodulator
        self.frequencies = numpy.asarray(frequencies)
        self.P0 = P0

    def _compute_U_inc(self, line):
        """
        Вычисляет амплитуду падающей волны по мощности P0 и волновому
        сопротивлению линии на текущей частоте.
        U_inc = sqrt(2 * P0 * Z0)
        """
        Z0 = line.Z0
        return numpy.sqrt(2 * self.P0 * Z0)

    def measure(self, line, defect, xc):
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
        features = []
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
            V = []
            for x in x_coords:
                if defect is None:
                    # Линия без дефекта – чисто бегущая волна
                    v = U_inc * numpy.exp(-1j * line.beta * x)
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

        return numpy.array(features)