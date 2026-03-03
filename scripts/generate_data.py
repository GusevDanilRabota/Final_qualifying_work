#!/usr/bin/env python3
"""
Скрипт для генерации синтетических данных измерений микрополосковой линии с дефектами.
"""

import argparse
import sys
import os

# Добавляем корневую директорию проекта в путь, чтобы импортировать src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import load_config, ensure_dirs
from src.data_generation import generate_data


def main():
    parser = argparse.ArgumentParser(description='Генерация синтетических данных')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Путь к файлу конфигурации YAML')
    args = parser.parse_args()

    # Загрузка конфигурации
    config = load_config(args.config)

    # Создание необходимых директорий
    ensure_dirs(config)

    # Генерация данных
    generate_data(config)


if __name__ == '__main__':
    main()