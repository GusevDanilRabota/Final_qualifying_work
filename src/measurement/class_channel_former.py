import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

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