import numpy

class microstrip_line_t:
    """
    Модель микрополосковой линии передачи.
    Позволяет рассчитать эффективную диэлектрическую проницаемость,
    волновое сопротивление и постоянную распространения с учётом частоты.
    """
    MU0 = 4 * numpy.pi * 1e-7   # Гн/м
    C0 = 299792458              # м/с

    def __init__(self, W, h, t, epsilon_r, length=None):
        """
        Параметры линии:
        W       – ширина полоска, м
        h       – высота подложки, м
        t       – толщина полоска, м
        epsilon_r – относительная диэлектрическая проницаемость подложки
        length  – физическая длина линии (опционально), м
        """
        self.W = W
        self.h = h
        self.t = t
        self.epsilon_r = epsilon_r
        self.length = length
        self.frequency = None          # текущая частота, Гц
        # Приватные поля для кэширования результатов расчётов
        self._W_eff = None
        self._epsilon_eff0 = None
        self._Z0_static = None
        self._fp = None
        self._m = None
        self._epsilon_eff = None
        self._Z0 = None
        self._beta = None

    def set_frequency(self, f):
        """Устанавливает рабочую частоту и пересчитывает все параметры."""
        self.frequency = f
        self._update_all()

    def _update_all(self):
        """Последовательный пересчёт всех параметров линии."""
        self._calc_W_eff()
        self._calc_epsilon_eff0()
        self._calc_Z0_static()
        self._calc_dispersion_params()
        self._calc_epsilon_eff()
        self._calc_Z0()
        self._calc_beta()

    def _calc_W_eff(self):
        """Эффективная ширина полоска с учётом толщины."""
        if self.W / self.h >= 0.1:
            self._W_eff = self.W + (self.t / numpy.pi) * (1 + numpy.log(2 * self.h / self.t))
        else:
            self._W_eff = self.W

    def _calc_epsilon_eff0(self):
        """Эффективная диэлектрическая проницаемость на нулевой частоте."""
        u = self._W_eff / self.h
        er = self.epsilon_r
        if u <= 1:
            self._epsilon_eff0 = ((er + 1)/2 + (er - 1)/2 * ((1 + 12/u)**(-0.5) + 0.04*(1 - u)**2))
        else:
            self._epsilon_eff0 = (er + 1)/2 + (er - 1)/2 * (1 + 12/u)**(-0.5)

    def _calc_Z0_static(self):
        """Волновое сопротивление на нулевой частоте."""
        u = self._W_eff / self.h
        eps_eff = self._epsilon_eff0
        if u <= 1:
            self._Z0_static = (60 / numpy.sqrt(eps_eff)) * numpy.log(8/u + u/4)
        else:
            self._Z0_static = (120 * numpy.pi) / (numpy.sqrt(eps_eff) * (u + 1.393 + 0.667 * numpy.log(u + 1.444)))

    def _calc_dispersion_params(self):
        """Параметры дисперсионной модели (частота fp и показатель m)."""
        u = self._W_eff / self.h
        self._fp = self._Z0_static / (2 * self.MU0 * self.h)
        self._m = 1.8 if u <= 1 else 2.0

    def _calc_epsilon_eff(self):
        """Эффективная проницаемость с учётом дисперсии."""
        if self.frequency is None:
            raise ValueError("Частота не установлена")
        f = self.frequency
        er = self.epsilon_r
        eps0 = self._epsilon_eff0
        fp = self._fp
        m = self._m
        self._epsilon_eff = er - (er - eps0) / (1 + (f / fp)**m)

    def _calc_Z0(self):
        """Волновое сопротивление с учётом дисперсии."""
        u = self._W_eff / self.h
        eps_eff = self._epsilon_eff
        if u <= 1:
            self._Z0 = (60 / numpy.sqrt(eps_eff)) * numpy.log(8/u + u/4)
        else:
            self._Z0 = (120 * numpy.pi) / (numpy.sqrt(eps_eff) * (u + 1.393 + 0.667 * numpy.log(u + 1.444)))

    def _calc_beta(self):
        """Постоянная распространения (рад/м)."""
        omega = 2 * numpy.pi * self.frequency
        self._beta = omega * numpy.sqrt(self._epsilon_eff) / self.C0

    # Свойства для доступа к рассчитанным величинам
    @property
    def W_eff(self):
        return self._W_eff

    @property
    def epsilon_eff0(self):
        return self._epsilon_eff0

    @property
    def Z0_static(self):
        return self._Z0_static

    @property
    def fp(self):
        return self._fp

    @property
    def epsilon_eff(self):
        return self._epsilon_eff

    @property
    def Z0(self):
        return self._Z0

    @property
    def beta(self):
        return self._beta

    @property
    def lambda_g(self):
        """Длина волны в линии (м)."""
        if self._beta is not None:
            return 2 * numpy.pi / self._beta
        return None


class defect_t(microstrip_line_t):
    """
    Модель локального неоднородного участка (дефекта) микрополосковой линии.
    Дефект представляет собой отрезок линии длиной L_def с изменёнными
    геометрическими или материальными параметрами.
    """
    def __init__(self, parent_line, W_def, h_def, t_def, epsilon_r_def,
                 x_d, L_def):
        """
        parent_line   – объект MicrostripLine для основной линии
        W_def, h_def, t_def, epsilon_r_def – параметры дефекта
        x_d           – координата центра дефекта вдоль линии (м)
        L_def         – длина дефекта (м)
        """
        super().__init__(W=W_def, h=h_def, t=t_def, epsilon_r=epsilon_r_def)
        self.parent_line = parent_line
        self.x_d = x_d
        self.L_def = L_def
        self.x1 = x_d - L_def / 2   # левая граница
        self.x2 = x_d + L_def / 2   # правая граница
        self._gamma = None           # комплексный коэффициент отражения
        self._T = None               # комплексный коэффициент прохождения

    def set_frequency(self, f):
        """
        Устанавливает частоту для дефекта и для родительской линии.
        Сбрасывает кэшированные gamma и T.
        """
        super().set_frequency(f)
        self.parent_line.set_frequency(f)
        self._gamma = None
        self._T = None

    def compute_gamma(self):
        """
        Вычисляет комплексный коэффициент отражения от дефекта.
        Используется модель отрезка линии с импедансом Z0_def,
        нагруженного на Z0 родительской линии.
        """
        Z0 = self.parent_line.Z0
        Z0_def = self.Z0
        beta_def = self.beta
        L = self.L_def

        tanBL = numpy.tan(beta_def * L)
        # Защита от больших значений тангенса (особенности тангенса при приближении к π/2)
        if numpy.isfinite(tanBL) and numpy.abs(tanBL) < 1e10:
            Z_in = Z0_def * (Z0 + 1j * Z0_def * tanBL) / (Z0_def + 1j * Z0 * tanBL)
        else:
            # Используем котангенс, если тангенс нестабилен
            cotBL = 1 / numpy.tan(beta_def * L) if numpy.abs(tanBL) > 1e10 else 0
            Z_in = Z0_def * (Z0 * cotBL + 1j * Z0_def) / (Z0_def * cotBL + 1j * Z0)

        gamma = (Z_in - Z0) / (Z_in + Z0)
        self._gamma = gamma
        return gamma

    @property
    def gamma(self):
        """Свойство для доступа к коэффициенту отражения (с ленивым вычислением)."""
        if self._gamma is None:
            self._gamma = self.compute_gamma()
        return self._gamma

    def compute_T(self):
        """
        Вычисляет комплексный коэффициент прохождения через дефект.
        """
        if self._gamma is None:
            self.compute_gamma()
        Z0 = self.parent_line.Z0
        Z0_def = self.Z0
        beta_def = self.beta
        beta_out = self.parent_line.beta
        L = self.L_def

        cosBL = numpy.cos(beta_def * L)
        sinBL = numpy.sin(beta_def * L)
        denominator = cosBL + 1j * (Z0_def / Z0) * sinBL
        T = (1 + self.gamma) * numpy.exp(1j * beta_out * self.x2) / denominator
        self._T = T
        return T

    @property
    def T(self):
        """Свойство для доступа к коэффициенту прохождения (ленивое вычисление)."""
        if self._T is None:
            self._T = self.compute_T()
        return self._T

    def voltage_at(self, x, U_inc):
        """
        Возвращает комплексное напряжение в точке x при падающей волне U_inc.
        Работает для точек слева, справа и внутри дефекта.
        """
        beta_out = self.parent_line.beta
        eps = 1e-12

        # Слева от дефекта (включая касание левой границы)
        if x <= self.x1 + eps:
            return U_inc * (numpy.exp(-1j * beta_out * x) + self.gamma * numpy.exp(-1j * beta_out * (2 * self.x1 - x)))

        # Справа от дефекта (включая касание правой границы)
        elif x >= self.x2 - eps:
            return U_inc * self.T * numpy.exp(-1j * beta_out * x)

        # Внутри дефекта
        else:
            # Напряжение на левой границе дефекта
            U_left = U_inc * (numpy.exp(-1j * beta_out * self.x1) + self.gamma * numpy.exp(-1j * beta_out * (2 * self.x1 - self.x1)))
            # Распространение внутри дефекта с его постоянной распространения
            return U_left * numpy.exp(-1j * self.beta * (x - self.x1))