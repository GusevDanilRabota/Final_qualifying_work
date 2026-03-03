#!/usr/bin/env python3
"""
Скрипт для обучения классификатора на сгенерированных данных.
"""

import argparse
import sys
import os
import logging
import pandas
import numpy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import load_config, ensure_dirs
from src.classification import train_and_evaluate
from src.visualization import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pca,
)
from src.utils import save_report


def setup_logging(level: str = "INFO") -> None:
    """Настройка логирования."""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    parser = argparse.ArgumentParser(description='Обучение модели классификации')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Путь к файлу конфигурации YAML')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Уровень логирования')
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Загрузка конфигурации
    try:
        config = load_config(args.config)
        logger.info(f"Конфигурация загружена из {args.config}")
    except Exception as e:
        logger.error(f"Ошибка загрузки конфигурации: {e}")
        sys.exit(1)

    # Создание директорий для отчётов и изображений
    ensure_dirs(config)

    # Загрузка данных
    data_path = config['paths']['data']
    if not os.path.exists(data_path):
        logger.error(f"Файл данных {data_path} не найден. Сначала выполните generate_data.py")
        sys.exit(1)

    df = pandas.read_csv(data_path)
    logger.info(f"Загружено {len(df)} записей из {data_path}")

    # Подготовка признаков
    feature_cols = [col for col in df.columns if col not in ['class', 'x_position']]
    X = df[feature_cols].values
    y = df['class'].values
    positions = df['x_position'].values

    # Обучение модели
    try:
        result = train_and_evaluate(config, X, y, positions)
        logger.info("Обучение модели завершено успешно")
    except Exception as e:
        logger.error(f"Ошибка при обучении модели: {e}")
        sys.exit(1)

    # Сохранение текстового отчёта
    report_path = os.path.join(config['paths']['reports'], 'classification_report.txt')
    save_report(result, config, feature_cols, report_path)

    # Визуализация матрицы ошибок
    classes = sorted(df['class'].unique())
    cm_path = os.path.join(config['paths']['figures'], 'confusion_matrix.png')
    plot_confusion_matrix(result['y_test'], result['y_pred'], classes,
                          save_path=cm_path)
    logger.info(f"Матрица ошибок сохранена в {cm_path}")

    # Важность признаков
    fi_path = os.path.join(config['paths']['figures'], 'feature_importance.png')
    plot_feature_importance(result['feature_importances'], feature_cols,
                            top_n=10, save_path=fi_path)
    logger.info(f"Важность признаков сохранена в {fi_path}")

    # PCA проекция
    pca_path = os.path.join(config['paths']['figures'], 'pca.png')
    colors = ['lightgray', 'red', 'green', 'blue', 'orange']
    labels = ['Нет дефекта', 'Утонение высоты', 'Утонение ширины',
              'Утонение подложки', 'Изменение εr']
    plot_pca(X, y, classes, colors, labels, save_path=pca_path)
    logger.info(f"PCA проекция сохранена в {pca_path}")

    logger.info("Обучение завершено. Результаты сохранены в директории reports/ и models/")


if __name__ == '__main__':
    main()