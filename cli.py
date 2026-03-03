#!/usr/bin/env python3
"""
Простой консольный интерфейс для управления проектом.
Позволяет запускать генерацию данных, обучение, оценку и визуализацию.
"""

import os
import sys
import subprocess
from typing import Optional

# Добавляем корневую папку проекта в путь для импорта модулей src
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.config import load_config


def print_header() -> None:
    """Выводит заголовок программы."""
    print("\n" + "=" * 60)
    print("     КЛАССИФИКАЦИЯ ДЕФЕКТОВ МИКРОПОЛОСКОВОЙ ЛИНИИ")
    print("=" * 60)


def print_menu() -> None:
    """Выводит главное меню."""
    print("\nГлавное меню:")
    print(" 1. Сгенерировать данные")
    print(" 2. Обучить модель")
    print(" 3. Оценить модель (на всех данных)")
    print(" 4. Визуализировать данные")
    print(" 5. Открыть папку с отчётами")
    print(" 6. Открыть папку с графиками")
    print(" 7. Изменить конфигурационный файл")
    print(" 0. Выход")
    print("-" * 40)


def run_script(script_name: str, config_path: str, extra_args: Optional[str] = None) -> bool:
    """
    Запускает указанный скрипт из папки scripts с переданным конфигом.

    Параметры
    ---------
    script_name : str
        Имя скрипта (например, 'generate_data.py').
    config_path : str
        Путь к конфигурационному файлу.
    extra_args : str, optional
        Дополнительные аргументы командной строки.

    Возвращает
    ----------
    bool
        True, если скрипт выполнен успешно, иначе False.
    """
    # Ищем скрипт в папке scripts
    script_path = os.path.join(os.path.dirname(__file__), "scripts", script_name)
    if not os.path.exists(script_path):
        print(f"Ошибка: скрипт {script_path} не найден.")
        return False

    cmd = [sys.executable, script_path, "--config", config_path]
    if extra_args:
        cmd.extend(extra_args.split())

    print(f"Запуск: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при выполнении скрипта: {e}")
        return False


def open_folder(path: str) -> None:
    """
    Открывает указанную папку в системном файловом менеджере (кроссплатформенно).

    Параметры
    ---------
    path : str
        Путь к папке.
    """
    if not os.path.exists(path):
        print(f"Папка {path} не существует.")
        return

    try:
        if sys.platform == 'win32':
            os.startfile(path)
        elif sys.platform == 'darwin':  # macOS
            subprocess.run(['open', path])
        else:  # Linux и другие Unix-подобные
            subprocess.run(['xdg-open', path])
        print(f"Открыта папка: {path}")
    except Exception as e:
        print(f"Не удалось открыть папку: {e}")


def main() -> None:
    """Основная функция интерфейса."""
    # Конфигурационный файл по умолчанию
    config_path = os.path.join(os.path.dirname(__file__), "config", "default.yaml")

    # Проверяем существование файла конфигурации
    if not os.path.exists(config_path):
        print(f"Предупреждение: файл конфигурации по умолчанию {config_path} не найден.")
        change = input("Указать другой путь? (y/n): ").strip().lower()
        if change == 'y':
            config_path = input("Введите путь к конфигурационному файлу: ").strip()
            if not os.path.exists(config_path):
                print("Файл не найден. Завершение.")
                return
        else:
            print("Работа без конфига невозможна. Завершение.")
            return

    # Главный цикл меню
    while True:
        print_header()
        print(f"Текущий конфиг: {config_path}")
        print_menu()
        choice = input("Выберите действие (0-7): ").strip()

        if choice == '1':
            run_script("generate_data.py", config_path)
        elif choice == '2':
            run_script("train_model.py", config_path)
        elif choice == '3':
            run_script("evaluate.py", config_path)
        elif choice == '4':
            run_script("visualize_data.py", config_path)
        elif choice == '5':
            try:
                config = load_config(config_path)
                reports_dir = config['paths']['reports']
                # Преобразуем относительный путь в абсолютный относительно корня проекта
                if not os.path.isabs(reports_dir):
                    reports_dir = os.path.join(os.path.dirname(__file__), reports_dir)
                open_folder(reports_dir)
            except Exception as e:
                print(f"Не удалось загрузить конфиг: {e}")
        elif choice == '6':
            try:
                config = load_config(config_path)
                figures_dir = config['paths']['figures']
                if not os.path.isabs(figures_dir):
                    figures_dir = os.path.join(os.path.dirname(__file__), figures_dir)
                open_folder(figures_dir)
            except Exception as e:
                print(f"Не удалось загрузить конфиг: {e}")
        elif choice == '7':
            new_path = input("Введите новый путь к конфигурационному файлу: ").strip()
            if os.path.exists(new_path):
                config_path = new_path
                print(f"Конфиг изменён на {config_path}")
            else:
                print("Файл не найден, конфиг не изменён.")
        elif choice == '0':
            print("Выход.")
            break
        else:
            print("Неверный ввод. Пожалуйста, выберите 0-7.")

        input("\nНажмите Enter, чтобы продолжить...")


if __name__ == "__main__":
    main()