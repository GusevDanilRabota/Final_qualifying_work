import os
import yaml
import numpy


def load_config(config_path):
    """
    Загружает конфигурацию из YAML-файла.

    Параметры
    ---------
    config_path : str
        Путь к файлу конфигурации.

    Возвращает
    ----------
    dict
        Словарь с параметрами конфигурации.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_frequencies_hz(config):
    """
    Извлекает массив частот в Гц из конфигурации.

    Параметры
    ---------
    config : dict
        Словарь конфигурации.

    Возвращает
    ----------
    numpy.ndarray
        Массив частот в Гц.
    """
    f_cfg = config['measurement']['frequencies']
    start_ghz = f_cfg['start']
    stop_ghz = f_cfg['stop']
    step_ghz = f_cfg['step']
    freqs_ghz = numpy.arange(start_ghz, stop_ghz + step_ghz/2, step_ghz)
    return freqs_ghz * 1e9


def get_feature_names(config):
    """
    Генерирует имена признаков на основе конфигурации.

    Параметры
    ---------
    config : dict
        Словарь конфигурации.

    Возвращает
    ----------
    list
        Список строк с именами признаков.
    """
    freqs_hz = get_frequencies_hz(config)
    freqs_ghz = freqs_hz / 1e9
    names = []
    for f in freqs_ghz:
        if f.is_integer():
            f_str = f"{int(f)}GHz"
        else:
            f_str = f"{f:.1f}GHz".replace('.', '_')
        for comp in ['I_S', 'Q_S', 'I_Dx', 'Q_Dx', 'I_Dy', 'Q_Dy']:
            names.append(f"{comp}_{f_str}")
    return names


def ensure_dirs(config):
    """
    Создаёт директории, указанные в разделе paths конфигурации,
    если они не существуют.

    Параметры
    ---------
    config : dict
        Словарь конфигурации.
    """
    paths = config.get('paths', {})
    for key, path in paths.items():
        if isinstance(path, str):
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Создана директория: {directory}")