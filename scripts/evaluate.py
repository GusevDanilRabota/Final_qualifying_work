#!/usr/bin/env python3
"""
Скрипт для оценки сохранённой модели на тестовых данных.
"""

import argparse
import sys
import os
import logging
import pandas
import numpy
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import load_config
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    parser = argparse.ArgumentParser(description='Оценка сохранённой модели')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Путь к файлу конфигурации YAML')
    parser.add_argument('--model', type=str, default=None,
                        help='Путь к файлу модели .pkl (если не указан, берётся из конфига)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Уровень логирования')
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Загрузка конфигурации
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Ошибка загрузки конфигурации: {e}")
        sys.exit(1)

    model_path = args.model if args.model else config['paths']['model']
    data_path = config['paths']['data']

    if not os.path.exists(model_path):
        logger.error(f"Файл модели {model_path} не найден.")
        sys.exit(1)
    if not os.path.exists(data_path):
        logger.error(f"Файл данных {data_path} не найден.")
        sys.exit(1)

    # Загрузка данных
    df = pandas.read_csv(data_path)
    feature_cols = [col for col in df.columns if col not in ['class', 'x_position']]
    X = df[feature_cols].values
    y = df['class'].values
    logger.info(f"Загружено {len(df)} записей")

    # Загрузка модели
    model = joblib.load(model_path)
    logger.info(f"Модель загружена из {model_path}")

    # Предсказание
    y_pred = model.predict(X)
    accuracy = numpy.mean(y_pred == y)
    logger.info(f"Accuracy на всех данных: {accuracy:.4f}")

    # Вывод метрик
    print("\n" + "="*60)
    print("ОЦЕНКА МОДЕЛИ НА ВСЕХ ДАННЫХ")
    print("="*60)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred))

    precision, recall, f1, support = precision_recall_fscore_support(y, y_pred)
    print("\nMacro averaged:")
    print(f"  Precision: {numpy.mean(precision):.4f}")
    print(f"  Recall:    {numpy.mean(recall):.4f}")
    print(f"  F1:        {numpy.mean(f1):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))


if __name__ == '__main__':
    main()