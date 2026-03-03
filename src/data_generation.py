import numpy
import pandas
import logging
from typing import List, Dict, Any
from tqdm import tqdm

from src.microstrip import microstrip_line_t, defect_t
from src.measurement import probe_t, channel_former_t, quadrature_demodulator_t, measurement_system_t

logger = logging.getLogger(__name__)


def generate_data(config: Dict[str, Any]) -> pandas.DataFrame:
    """
    Генерирует синтетические данные измерений для микрополосковой линии с дефектами.

    Параметры
    ---------
    config : dict
        Словарь конфигурации, загруженный из YAML-файла.
        Ожидается структура, описанная в config/default.yaml.

    Возвращает
    ----------
    pandas.DataFrame
        Таблица с признаками (по 6 на каждую частоту: I_S, Q_S, I_Dx, Q_Dx, I_Dy, Q_Dy),
        столбцом 'class' (метка класса: 0 – нет дефекта, 1..4 – типы дефектов)
        и столбцом 'x_position' (координата центра головки).
    """
    # --- Инициализация основной линии ---
    line_params = config['line']
    parent = microstrip_line_t(
        W=line_params['width'],
        h=line_params['height'],
        t=line_params['thickness'],
        epsilon_r=line_params['epsilon_r'],
        length=line_params.get('length')
    )
    logger.info("Основная линия создана")

    # --- Параметры дефектов из конфига ---
    def_cfg = config['defects']
    x_centers = def_cfg['positions']
    L_def = def_cfg['length']
    types = def_cfg['types']
    tp = def_cfg['type_params']

    # Вычисление параметров каждого типа дефекта
    t1 = line_params['thickness'] * tp['t1_factor']          # утонение высоты (тип 1)
    W2 = line_params['width'] * tp['W2_factor']              # утонение ширины (тип 2)
    h3 = line_params['height'] * tp['h3_factor']             # утонение подложки (тип 3)
    eps4 = line_params['epsilon_r'] * tp['eps4_factor']      # изменение εr (тип 4)

    # Создание объектов дефектов
    defects = [
        defect_t(parent, line_params['width'], line_params['height'], t1, line_params['epsilon_r'], x_centers[0], L_def),
        defect_t(parent, W2, line_params['height'], line_params['thickness'], line_params['epsilon_r'], x_centers[1], L_def),
        defect_t(parent, line_params['width'], h3, line_params['thickness'], line_params['epsilon_r'], x_centers[2], L_def),
        defect_t(parent, line_params['width'], line_params['height'], line_params['thickness'], eps4, x_centers[3], L_def)
    ]
    logger.info(f"Создано {len(defects)} дефектов")

    # --- Измерительная система ---
    probe = probe_t(config['probe']['a'])

    # Частоты: из конфига заданы в ГГц, переводим в Гц
    f_start = config['measurement']['frequencies']['start'] * 1e9
    f_stop  = config['measurement']['frequencies']['stop'] * 1e9
    f_step  = config['measurement']['frequencies']['step'] * 1e9
    frequencies = numpy.arange(f_start, f_stop + f_step/2, f_step)

    demod = quadrature_demodulator_t(snr_db=config['measurement']['snr_db'])
    channel_former = channel_former_t()
    ms = measurement_system_t(probe, channel_former, demod, frequencies, P0=config['measurement']['P0'])
    logger.info("Измерительная система создана")

    # --- Позиции сканирования ---
    L_line = line_params['length']
    a = probe.a
    x_start = a * config['scan']['x_start_offset']
    x_stop = L_line - a * config['scan']['x_start_offset']
    step = config['scan']['x_step']
    x_positions = numpy.arange(x_start, x_stop + step/2, step)
    repeats = config['scan']['repeats_per_position']
    logger.info(f"Позиций сканирования: {len(x_positions)}, повторений: {repeats}")

    # --- Генерация данных ---
    all_data: List[List[float]] = []
    tol = 1e-12

    # Проверка наличия параллельной обработки (можно реализовать позже)
    parallel_enabled = config.get('parallel', {}).get('enabled', False)
    if parallel_enabled:
        logger.warning("Параллельная обработка пока не реализована, используется последовательный режим")

    for defect, typ in zip(defects, types):
        logger.info(f"Генерация для дефекта типа {typ}")
        for xc in tqdm(x_positions, desc=f'Тип {typ}'):
            x_elec = probe.get_electrode_coords(xc)
            x1, x2 = defect.x1, defect.x2

            all_left = all(x <= x1 + tol for x in x_elec)
            all_right = all(x >= x2 - tol for x in x_elec)
            all_inside = all(x1 + tol <= x <= x2 - tol for x in x_elec)

            if all_left or all_right:
                label = 0
            elif all_inside:
                label = typ
            else:
                continue

            for rep in range(repeats):
                features = ms.measure(parent, defect, xc)
                row = numpy.concatenate([features, [label, xc]])
                all_data.append(row)

    logger.info(f"Сгенерировано {len(all_data)} строк")

    # --- Формирование DataFrame ---
    freq_ghz = frequencies / 1e9
    feature_names = []
    for f in freq_ghz:
        f_str = f"{int(f)}GHz" if f.is_integer() else f"{f:.1f}GHz".replace('.', '_')
        for comp in ['I_S', 'Q_S', 'I_Dx', 'Q_Dx', 'I_Dy', 'Q_Dy']:
            feature_names.append(f"{comp}_{f_str}")
    feature_names.append('class')
    feature_names.append('x_position')

    df = pandas.DataFrame(all_data, columns=feature_names)

    output_path = config['paths']['data']
    df.to_csv(output_path, index=False)
    logger.info(f"Данные сохранены в {output_path}")

    return df