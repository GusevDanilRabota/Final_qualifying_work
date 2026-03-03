import numpy
import logging
from typing import Optional

from .class_microstrip_line import microstrip_line_t

# Настройка логгера для модуля
logger = logging.getLogger(__name__)

class defect_t(microstrip_line_t):
    """
    Модель локального неоднородного участка (дефекта) микрополосковой линии.
    Дефект представляет собой отрезок линии длиной L_def с изменёнными
    геометрическими или материальными параметрами.
    """
    def __init__(self, parent_line: microstrip_line_t, W_def: float, h_def: float,
                 t_def: float, epsilon_r_def: float, x_d: float, L_def: float) -> None:
        """
        parent_line   – объект microstrip_line_t для основной линии
        W_def, h_def, t_def, epsilon_r_def – параметры дефекта
        x_d           – координата центра дефекта вдоль линии (м)
        L_def         – длина дефекта (м)
        """
        super().__init__(W=W_def, h=h_def, t=t_def, epsilon_r=epsilon_r_def)
        # Проверки
        if x_d < 0 or L_def <= 0:
            raise ValueError("Координата центра дефекта должна быть неотрицательной, длина дефекта положительной")
        if parent_line.length is not None and (x_d - L_def/2 < 0 or x_d + L_def/2 > parent_line.length):
            logger.warning("Дефект выходит за границы линии")

        self.parent_line: microstrip_line_t = parent_line
        self.x_d: float = x_d
        self.L_def: float = L_def
        self.x1: float = x_d - L_def / 2   # левая граница
        self.x2: float = x_d + L_def / 2   # правая граница
        self._gamma: Optional[complex] = None
        self._T: Optional[complex] = None

    def set_frequency(self, f: float) -> None:
        """
        Устанавливает частоту для дефекта и для родительской линии.
        Сбрасывает кэшированные gamma и T.
        """
        super().set_frequency(f)
        self.parent_line.set_frequency(f)
        self._gamma = None
        self._T = None
        logger.debug(f"Частота дефекта установлена: {f/1e9:.3f} ГГц")

    def compute_gamma(self) -> complex:
        """
        Вычисляет комплексный коэффициент отражения от дефекта.
        Используется модель отрезка линии с импедансом Z0_def,
        нагруженного на Z0 родительской линии.
        """
        Z0 = self.parent_line.Z0
        Z0_def = self.Z0
        beta_def = self.beta
        L = self.L_def

        # Проверка, что все необходимые параметры вычислены
        if Z0 is None or Z0_def is None or beta_def is None:
            raise RuntimeError("Не удалось получить параметры линии. Установите частоту.")

        try:
            tanBL = numpy.tan(beta_def * L)
            # Защита от больших значений тангенса (особенности при приближении к π/2)
            if numpy.isfinite(tanBL) and numpy.abs(tanBL) < 1e10:
                Z_in = Z0_def * (Z0 + 1j * Z0_def * tanBL) / (Z0_def + 1j * Z0 * tanBL)
            else:
                # Используем котангенс, если тангенс нестабилен
                cotBL = 1 / numpy.tan(beta_def * L) if numpy.abs(tanBL) > 1e10 else 0
                Z_in = Z0_def * (Z0 * cotBL + 1j * Z0_def) / (Z0_def * cotBL + 1j * Z0)
        except Exception as e:
            logger.error(f"Ошибка вычисления входного импеданса дефекта: {e}")
            raise

        gamma = (Z_in - Z0) / (Z_in + Z0)
        self._gamma = gamma
        logger.debug(f"gamma = {gamma:.4f}")
        return gamma

    @property
    def gamma(self) -> complex:
        """Свойство для доступа к коэффициенту отражения (с ленивым вычислением)."""
        if self._gamma is None:
            self._gamma = self.compute_gamma()
        return self._gamma

    def compute_T(self) -> complex:
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

        if Z0 is None or Z0_def is None or beta_def is None or beta_out is None:
            raise RuntimeError("Не удалось получить параметры линии. Установите частоту.")

        try:
            cosBL = numpy.cos(beta_def * L)
            sinBL = numpy.sin(beta_def * L)
            denominator = cosBL + 1j * (Z0_def / Z0) * sinBL
            T = (1 + self.gamma) * numpy.exp(1j * beta_out * self.x2) / denominator
        except Exception as e:
            logger.error(f"Ошибка вычисления коэффициента прохождения: {e}")
            raise

        self._T = T
        logger.debug(f"T = {T:.4f}")
        return T

    @property
    def T(self) -> complex:
        """Свойство для доступа к коэффициенту прохождения (ленивое вычисление)."""
        if self._T is None:
            self._T = self.compute_T()
        return self._T

    def voltage_at(self, x: float, U_inc: complex) -> complex:
        """
        Возвращает комплексное напряжение в точке x при падающей волне U_inc.
        Работает для точек слева, справа и внутри дефекта.
        """
        beta_out = self.parent_line.beta
        if beta_out is None or self.beta is None:
            raise RuntimeError("Не установлена частота для линии или дефекта")

        eps = 1e-12

        # Слева от дефекта (включая касание левой границы)
        if x <= self.x1 + eps:
            return U_inc * (numpy.exp(-1j * beta_out * x) +
                            self.gamma * numpy.exp(-1j * beta_out * (2 * self.x1 - x)))

        # Справа от дефекта (включая касание правой границы)
        elif x >= self.x2 - eps:
            return U_inc * self.T * numpy.exp(-1j * beta_out * x)

        # Внутри дефекта
        else:
            # Напряжение на левой границе дефекта
            U_left = U_inc * (numpy.exp(-1j * beta_out * self.x1) +
                              self.gamma * numpy.exp(-1j * beta_out * (2 * self.x1 - self.x1)))
            # Распространение внутри дефекта с его постоянной распространения
            return U_left * numpy.exp(-1j * self.beta * (x - self.x1))