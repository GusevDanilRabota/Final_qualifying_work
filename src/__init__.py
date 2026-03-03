"""
Пакет для моделирования микрополосковой линии с дефектами,
генерации синтетических данных и классификации типов дефектов.
"""

# Классы микрополосковой линии и дефектов
from .microstrip import microstrip_line_t, defect_t

# Классы измерительной системы
from .measurement import probe_t, channel_former_t, quadrature_demodulator_t, measurement_system_t

# Генерация данных
from .data_generation import generate_data

# Обучение и оценка модели
from .classification import train_and_evaluate

# Функции визуализации
from .visualization import (
    plot_defect_map,
    plot_hodographs,
    plot_per_frequency_scatter,
    plot_pca,
    plot_frequency_dependence,
    plot_phase_frequency,
    plot_feature_importance,
    plot_tree_fragment,
    plot_confusion_matrix,
    plot_comparison_map
)

# Работа с конфигурацией
from .config import load_config, get_frequencies_hz, get_feature_names, ensure_dirs

# Утилиты (сохранение отчёта)
from .utils import save_report

__all__ = [
    # microstrip
    'microstrip_line_t',
    'defect_t',
    # measurement
    'probe_t',
    'channel_former_t',
    'quadrature_demodulator_t',
    'measurement_system_t',
    # data_generation
    'generate_data',
    # classification
    'train_and_evaluate',
    # visualization
    'plot_defect_map',
    'plot_hodographs',
    'plot_per_frequency_scatter',
    'plot_pca',
    'plot_frequency_dependence',
    'plot_phase_frequency',
    'plot_feature_importance',
    'plot_tree_fragment',
    'plot_confusion_matrix',
    'plot_comparison_map',
    # config
    'load_config',
    'get_frequencies_hz',
    'get_feature_names',
    'ensure_dirs',
    # utils
    'save_report'
]