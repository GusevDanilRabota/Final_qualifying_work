import numpy
import logging
from typing import Optional

# Настройка логгера для модуля
logger = logging.getLogger(__name__)


class microstrip_line_t:
    """
    Модель микрополосковой линии передачи.
    Позволяет рассчитать эффективную диэлектрическую проницаемость,
    волновое сопротивление и постоянную распространения с учётом частоты.
    """
    MU0: float = 4 * numpy.pi * 1e-7    # Гн/м
    C0: float = 299792458               # м/с

    def __init__(self, W: float, h: float, t: float, epsilon_r: float, length: Optional[float] = None) -> None:
        """
        Параметры линии:
        W       – ширина полоска, м
        h       – высота подложки, м
        t       – толщина полоска, м
        epsilon_r – относительная диэлектрическая проницаемость подложки
        length  – физическая длина линии (опционально), м
        """
        # Проверка входных данных
        if W <= 0 or h <= 0 or t <= 0 or epsilon_r <= 0:
            raise ValueError("Все геометрические размеры и диэлектрическая проницаемость должны быть положительными")
        if length is not None and length <= 0:
            raise ValueError("Длина линии должна быть положительной")

        self.W: float = W
        self.h: float = h
        self.t: float = t
        self.epsilon_r: float = epsilon_r
        self.length: Optional[float] = length
        self.frequency: Optional[float] = None          # текущая частота, Гц

        # Приватные поля для кэширования результатов расчётов
        self._W_eff: Optional[float] = None
        self._epsilon_eff0: Optional[float] = None
        self._Z0_static: Optional[float] = None
        self._fp: Optional[float] = None
        self._m: Optional[float] = None
        self._epsilon_eff: Optional[float] = None
        self._Z0: Optional[float] = None
        self._beta: Optional[float] = None

    def set_frequency(self, f: float) -> None:
        """Устанавливает рабочую частоту и пересчитывает все параметры."""
        if f <= 0:
            raise ValueError("Частота должна быть положительной")
        self.frequency = f
        self._update_all()
        logger.debug(f"Частота установлена: {f/1e9:.3f} ГГц")

    def _update_all(self) -> None:
        """Последовательный пересчёт всех параметров линии."""
        self._calc_W_eff()
        self._calc_epsilon_eff0()
        self._calc_Z0_static()
        self._calc_dispersion_params()
        self._calc_epsilon_eff()
        self._calc_Z0()
        self._calc_beta()

    def _calc_W_eff(self) -> None:
        """Эффективная ширина полоска с учётом толщины."""
        if self.W / self.h >= 0.1:
            self._W_eff = self.W + (self.t / numpy.pi) * (1 + numpy.log(2 * self.h / self.t))
        else:
            self._W_eff = self.W
        logger.debug(f"W_eff = {self._W_eff:.3e} м")

    def _calc_epsilon_eff0(self) -> None:
        """Эффективная диэлектрическая проницаемость на нулевой частоте."""
        u = self._W_eff / self.h
        er = self.epsilon_r
        if u <= 1:
            self._epsilon_eff0 = ((er + 1)/2 + (er - 1)/2 * ((1 + 12/u)**(-0.5) + 0.04*(1 - u)**2))
        else:
            self._epsilon_eff0 = (er + 1)/2 + (er - 1)/2 * (1 + 12/u)**(-0.5)
        logger.debug(f"epsilon_eff0 = {self._epsilon_eff0:.4f}")

    def _calc_Z0_static(self) -> None:
        """Волновое сопротивление на нулевой частоте."""
        u = self._W_eff / self.h
        eps_eff = self._epsilon_eff0
        try:
            if u <= 1:
                self._Z0_static = (60 / numpy.sqrt(eps_eff)) * numpy.log(8/u + u/4)
            else:
                self._Z0_static = (120 * numpy.pi) / (numpy.sqrt(eps_eff) * (u + 1.393 + 0.667 * numpy.log(u + 1.444)))
        except Exception as e:
            logger.error(f"Ошибка вычисления Z0_static: {e}")
            raise
        logger.debug(f"Z0_static = {self._Z0_static:.3f} Ом")

    def _calc_dispersion_params(self) -> None:
        """Параметры дисперсионной модели (частота fp и показатель m)."""
        u = self._W_eff / self.h
        self._fp = self._Z0_static / (2 * self.MU0 * self.h)
        self._m = 1.8 if u <= 1 else 2.0
        logger.debug(f"fp = {self._fp/1e9:.3f} ГГц, m = {self._m}")

    def _calc_epsilon_eff(self) -> None:
        """Эффективная проницаемость с учётом дисперсии."""
        if self.frequency is None:
            raise ValueError("Частота не установлена. Вызовите set_frequency() перед расчётом.")
        f = self.frequency
        er = self.epsilon_r
        eps0 = self._epsilon_eff0
        fp = self._fp
        m = self._m
        self._epsilon_eff = er - (er - eps0) / (1 + (f / fp)**m)
        logger.debug(f"epsilon_eff = {self._epsilon_eff:.4f}")

    def _calc_Z0(self) -> None:
        """Волновое сопротивление с учётом дисперсии."""
        u = self._W_eff / self.h
        eps_eff = self._epsilon_eff
        try:
            if u <= 1:
                self._Z0 = (60 / numpy.sqrt(eps_eff)) * numpy.log(8/u + u/4)
            else:
                self._Z0 = (120 * numpy.pi) / (numpy.sqrt(eps_eff) * (u + 1.393 + 0.667 * numpy.log(u + 1.444)))
        except Exception as e:
            logger.error(f"Ошибка вычисления Z0: {e}")
            raise
        logger.debug(f"Z0 = {self._Z0:.3f} Ом")

    def _calc_beta(self) -> None:
        """Постоянная распространения (рад/м)."""
        omega = 2 * numpy.pi * self.frequency
        self._beta = omega * numpy.sqrt(self._epsilon_eff) / self.C0
        logger.debug(f"beta = {self._beta:.3f} рад/м")

    # Свойства для доступа к рассчитанным величинам
    @property
    def W_eff(self) -> Optional[float]:
        return self._W_eff

    @property
    def epsilon_eff0(self) -> Optional[float]:
        return self._epsilon_eff0

    @property
    def Z0_static(self) -> Optional[float]:
        return self._Z0_static

    @property
    def fp(self) -> Optional[float]:
        return self._fp

    @property
    def epsilon_eff(self) -> Optional[float]:
        return self._epsilon_eff

    @property
    def Z0(self) -> Optional[float]:
        return self._Z0

    @property
    def beta(self) -> Optional[float]:
        return self._beta

    @property
    def lambda_g(self) -> Optional[float]:
        """Длина волны в линии (м)."""
        if self._beta is not None:
            return 2 * numpy.pi / self._beta
        return None