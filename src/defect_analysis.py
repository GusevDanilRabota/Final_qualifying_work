import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

def analyze_hodograph(I_values: np.ndarray, Q_values: np.ndarray, frequencies: np.ndarray) -> str:
    """
    Анализирует годограф (I, Q) на нескольких частотах и определяет характер дефекта.

    Параметры
    ---------
    I_values : np.ndarray, shape (n_freqs, n_samples) or (n_freqs,)
        Значения I для каждой частоты (можно усреднённые).
    Q_values : np.ndarray, shape (n_freqs, n_samples) or (n_freqs,)
        Значения Q для каждой частоты.
    frequencies : np.ndarray, shape (n_freqs,)
        Частоты в Гц.

    Возвращает
    ----------
    str
        Описание характера дефекта: "Индуктивный", "Ёмкостной", "Резистивный", "Неопределённо".
    """
    # Приведение к размерности (n_freqs,)
    if I_values.ndim == 2:
        I_mean = np.mean(I_values, axis=1)
        Q_mean = np.mean(Q_values, axis=1)
    else:
        I_mean = I_values
        Q_mean = Q_values

    # Вычисляем фазу (в радианах) для каждой частоты
    phase = np.angle(I_mean + 1j * Q_mean)
    # Разворачиваем фазу, чтобы избежать скачков
    phase_unwrapped = np.unwrap(phase)

    # Аппроксимируем линейную зависимость фазы от частоты
    if len(frequencies) < 2:
        return "Недостаточно частот для анализа"

    coeffs = np.polyfit(frequencies, phase_unwrapped, 1)
    slope = coeffs[0]  # рад/Гц

    logger.debug(f"Наклон фазы: {slope:.3e} рад/Гц")

    threshold = 1e-12  # порог для определения значимости наклона

    if abs(slope) < threshold:
        return "Резистивный (чисто активный)"
    elif slope > 0:
        return "Индуктивный"
    else:
        return "Ёмкостной"


def analyze_from_dataframe(df, freq_cols, class_label=None, x_position=None):
    """
    Анализирует данные из DataFrame.

    Параметры
    ---------
    df : pandas.DataFrame
        Данные с признаками I_Dx_* и Q_Dx_*.
    freq_cols : list
        Список строк с названиями частот (например, ['1GHz', '2GHz', ...]).
    class_label : int, optional
        Если задан, анализируются только записи с данным классом (усреднение по всем позициям и повторам).
    x_position : float, optional
        Если задан, анализируются только записи с данной координатой (усреднение по повторам).

    Возвращает
    ----------
    str
        Результат анализа.
    """
    if class_label is not None:
        subset = df[df['class'] == class_label]
        if len(subset) == 0:
            return f"Нет данных для класса {class_label}"
    elif x_position is not None:
        subset = df[np.isclose(df['x_position'], x_position)]
        if len(subset) == 0:
            return f"Нет данных для позиции x = {x_position}"
    else:
        return "Не указан фильтр (класс или позиция)"

    I_values = []
    Q_values = []
    for f in freq_cols:
        I_col = f'I_Dx_{f}'
        Q_col = f'Q_Dx_{f}'
        if I_col not in subset.columns or Q_col not in subset.columns:
            return f"Столбцы для частоты {f} не найдены"
        I_values.append(subset[I_col].values)
        Q_values.append(subset[Q_col].values)

    I_array = np.array(I_values)  # shape (n_freqs, n_samples)
    Q_array = np.array(Q_values)

    # Частоты в Гц (предполагаем, что f задано в ГГц)
    freqs_ghz = [float(f.replace('GHz', '')) for f in freq_cols]
    freqs_hz = np.array(freqs_ghz) * 1e9

    return analyze_hodograph(I_array, Q_array, freqs_hz)