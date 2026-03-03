#!/usr/bin/env python3
"""
Скрипт для визуализации сгенерированных данных без обучения модели.
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import load_config
from src.visualization import (
    plot_hodographs,
    plot_per_frequency_scatter,
    plot_pca,
    plot_frequency_dependence,
    plot_phase_frequency,
    plot_defect_map
)
from src.microstrip import microstrip_line_t, defect_t


def main():
    parser = argparse.ArgumentParser(description='Визуализация данных')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Путь к файлу конфигурации YAML')
    parser.add_argument('--data', type=str, default=None,
                        help='Путь к CSV-файлу с данными (если не указан, берётся из конфига)')
    args = parser.parse_args()

    config = load_config(args.config)
    data_path = args.data if args.data else config['paths']['data']

    if not os.path.exists(data_path):
        print(f"Ошибка: файл данных {data_path} не найден.")
        return

    df = pd.read_csv(data_path)
    print(f"Загружено {len(df)} записей из {data_path}")

    # Определяем доступные частоты из названий столбцов
    freqs = sorted(set([int(col.split('_')[-1].replace('GHz', ''))
                        for col in df.columns if col.endswith('GHz')]))

    classes = sorted(df['class'].unique())
    colors = ['lightgray', 'red', 'green', 'blue', 'orange']
    labels = ['Нет дефекта', 'Утонение высоты', 'Утонение ширины',
              'Утонение подложки', 'Изменение εr']

    # Построение всех графиков
    print("Построение годографов для частоты 5 ГГц...")
    plot_hodographs(df, 5, classes, colors, labels,
                    save_path=os.path.join(config['paths']['figures'], 'hodographs.png'))

    print("Построение многопанельного scatter...")
    plot_per_frequency_scatter(df, freqs, classes, colors, labels,
                               save_path=os.path.join(config['paths']['figures'], 'per_frequency_scatter.png'))

    print("PCA проекция...")
    feature_cols = [col for col in df.columns if col not in ['class', 'x_position']]
    X = df[feature_cols].values
    y = df['class'].values
    plot_pca(X, y, classes, colors, labels,
             save_path=os.path.join(config['paths']['figures'], 'pca.png'))

    print("Частотная зависимость I_S...")
    plot_frequency_dependence(df, freqs, classes, colors, labels, channel='I_S',
                              save_path=os.path.join(config['paths']['figures'], 'frequency_dependence.png'))

    print("Частотная зависимость фазы Dx...")
    plot_phase_frequency(df, freqs, classes, colors, labels, channel='Dx',
                         save_path=os.path.join(config['paths']['figures'], 'phase_frequency.png'))

    # Карта дефектов (требуется воссоздать объекты Defect из конфига)
    line_params = config['line']
    parent = microstrip_line_t(
        W=line_params['width'],
        h=line_params['height'],
        t=line_params['thickness'],
        epsilon_r=line_params['epsilon_r']
    )
    def_cfg = config['defects']
    x_centers = def_cfg['positions']
    L_def = def_cfg['length']
    types = def_cfg['types']
    tp = def_cfg['type_params']

    t1 = line_params['thickness'] * tp['t1_factor']
    W2 = line_params['width'] * tp['W2_factor']
    h3 = line_params['height'] * tp['h3_factor']
    eps4 = line_params['epsilon_r'] * tp['eps4_factor']

    defects = [
        defect_t(parent, line_params['width'], line_params['height'], t1, line_params['epsilon_r'], x_centers[0], L_def),
        defect_t(parent, W2, line_params['height'], line_params['thickness'], line_params['epsilon_r'], x_centers[1], L_def),
        defect_t(parent, line_params['width'], h3, line_params['thickness'], line_params['epsilon_r'], x_centers[2], L_def),
        defect_t(parent, line_params['width'], line_params['height'], line_params['thickness'], eps4, x_centers[3], L_def)
    ]

    plot_defect_map(defects, types, line_params['length'], line_params['width'],
                    colors, labels,
                    save_path=os.path.join(config['paths']['figures'], 'defect_map.png'))

    print(f"Все графики сохранены в {config['paths']['figures']}")


if __name__ == '__main__':
    main()