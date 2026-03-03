#!/usr/bin/env python3
"""
Скрипт для оценки сохранённой модели на тестовых данных.
"""

import argparse
import sys
import os
import pandas
import numpy
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import load_config
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support


def main():
    parser = argparse.ArgumentParser(description='Оценка сохранённой модели')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Путь к файлу конфигурации YAML')
    parser.add_argument('--model', type=str, default=None,
                        help='Путь к файлу модели .pkl (если не указан, берётся из конфига)')
    args = parser.parse_args()

    # Загрузка конфигурации
    config = load_config(args.config)
    model_path = args.model if args.model else config['paths']['model']
    data_path = config['paths']['data']

    if not os.path.exists(model_path):
        print(f"Ошибка: файл модели {model_path} не найден.")
        return
    if not os.path.exists(data_path):
        print(f"Ошибка: файл данных {data_path} не найден.")
        return

    # Загрузка данных
    df = pandas.read_csv(data_path)
    feature_cols = [col for col in df.columns if col not in ['class', 'x_position']]
    X = df[feature_cols].values
    y = df['class'].values

    # Загрузка модели
    model = joblib.load(model_path)

    # Предсказание
    y_pred = model.predict(X)

    # Вывод метрик
    print("\n" + "="*60)
    print("ОЦЕНКА МОДЕЛИ НА ВСЕХ ДАННЫХ")
    print("="*60)
    print(f"Accuracy: {numpy.mean(y_pred == y):.4f}")
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