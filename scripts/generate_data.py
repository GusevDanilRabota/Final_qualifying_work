#!/usr/bin/env python3
"""
Скрипт для генерации синтетических данных измерений микрополосковой линии с дефектами.
"""

import argparse
import sys
import os
import logging

# Добавляем корневую директорию проекта в путь, чтобы импортировать src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import load_config, ensure_dirs
from src.data_generation import generate_data


def setup_logging(level: str = "INFO") -> None:
    """Настройка логирования."""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    parser = argparse.ArgumentParser(description='Генерация синтетических данных')
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

    # Создание необходимых директорий
    ensure_dirs(config)
    logger.info("Директории созданы/проверены")

    # Генерация данных
    try:
        df = generate_data(config)
        logger.info(f"Данные успешно сгенерированы: {df.shape[0]} строк")
    except Exception as e:
        logger.error(f"Ошибка при генерации данных: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()