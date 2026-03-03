#!/usr/bin/env python3
"""
Скрипт для обучения классификатора на сгенерированных данных.
"""

import argparse
import sys
import os
import pandas
import numpy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import load_config, ensure_dirs, get_feature_names
from src.classification import train_and_evaluate
from src.visualization import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pca,
)
from src.utils import save_report


def main():
    parser = argparse.ArgumentParser(description='Обучение модели классификации')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Путь к файлу конфигурации YAML')
    args = parser.parse_args()

    # Загрузка конфигурации
    config = load_config(args.config)

    # Создание директорий для отчётов и изображений
    ensure_dirs(config)

    # Загрузка данных
    data_path = config['paths']['data']
    if not os.path.exists(data_path):
        print(f"Ошибка: файл данных {data_path} не найден. Сначала выполните generate_data.py")
        return

    df = pandas.read_csv(data_path)
    print(f"Загружено {len(df)} записей из {data_path}")

    # Подготовка признаков
    feature_cols = [col for col in df.columns if col not in ['class', 'x_position']]
    X = df[feature_cols].values
    y = df['class'].values
    positions = df['x_position'].values

    # Обучение модели
    result = train_and_evaluate(config, X, y, positions)

    # Сохранение текстового отчёта
    report_path = os.path.join(config['paths']['reports'], 'classification_report.txt')
    save_report(result, config, feature_cols, report_path)

    # Визуализация матрицы ошибок
    classes = sorted(df['class'].unique())
    cm_path = os.path.join(config['paths']['figures'], 'confusion_matrix.png')
    plot_confusion_matrix(result['y_test'], result['y_pred'], classes,
                          save_path=cm_path)

    # Важность признаков
    fi_path = os.path.join(config['paths']['figures'], 'feature_importance.png')
    plot_feature_importance(result['feature_importances'], feature_cols,
                            top_n=10, save_path=fi_path)

    # PCA проекция
    pca_path = os.path.join(config['paths']['figures'], 'pca.png')
    colors = ['lightgray', 'red', 'green', 'blue', 'orange']
    labels = ['Нет дефекта', 'Утонение высоты', 'Утонение ширины',
              'Утонение подложки', 'Изменение εr']
    plot_pca(X, y, classes, colors, labels, save_path=pca_path)

    print("Обучение завершено. Результаты сохранены в директории reports/ и models/")


if __name__ == '__main__':
    main()