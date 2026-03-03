import logging
from typing import List

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