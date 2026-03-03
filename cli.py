#!/usr/bin/env python3
"""
Улучшенный консольный интерфейс с максимальным набором функций.
Поддерживает режимы пользователя и администратора, интерактивное редактирование конфигурации,
расширенную статистику, экспорт в HTML, контекстную справку и многое другое.
"""

import os
import sys
import subprocess
import logging
import getpass
import shutil
import time
from datetime import datetime
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
from colorama import init, Fore, Back, Style

# Добавляем корневую папку проекта в путь для импорта модулей src
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Проверка наличия обязательных зависимостей
try:
    import yaml
    import joblib
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    print(Fore.RED + f"Ошибка: отсутствует необходимая библиотека: {e}")
    print(Fore.YELLOW + "Установите зависимости: pip install -r requirements.txt")
    sys.exit(1)

from src.config import load_config, ensure_dirs, get_feature_names
from src.data_generation import generate_data
from src.classification import train_and_evaluate
from src.visualization import (
    plot_confusion_matrix, plot_feature_importance, plot_pca,
    plot_hodographs, plot_per_frequency_scatter, plot_frequency_dependence,
    plot_phase_frequency, plot_defect_map, plot_comparison_map
)
from src.utils import save_report
from src.microstrip import microstrip_line_t, defect_t
from src.defect_analysis import analyze_from_dataframe

init(autoreset=True)

# Информация о программе
AUTHOR = "Данил Гусев"
LICENSE = "MIT"
VERSION = "3.0.0"
ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD', 'admin123')

# Файл для записи истории команд
HISTORY_FILE = os.path.join("reports", "command_history.log")
os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)


def log_action(action: str):
    """Записывает действие в историю с временной меткой."""
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} - {action}\n")


def print_header(start_time: str, mode: str = "ГЛАВНОЕ МЕНЮ", config_path: str = "") -> None:
    """Выводит заголовок программы с указанием режима и текущего конфига."""
    print(Fore.CYAN + "\n" + "╔" + "═" * 58 + "╗")
    print(Fore.CYAN + "║" + Fore.YELLOW + "{:^58}".format("КЛАССИФИКАЦИЯ ДЕФЕКТОВ МИКРОПОЛОСКОВОЙ ЛИНИИ") + Fore.CYAN + "║")
    print(Fore.CYAN + "╠" + "═" * 58 + "╣")
    print(Fore.CYAN + "║ Автор: {:<48} ║".format(AUTHOR))
    print(Fore.CYAN + "║ Лицензия: {:<44} ║".format(LICENSE))
    print(Fore.CYAN + "║ Версия: {:<46} ║".format(VERSION))
    print(Fore.CYAN + "║ Запущено: {:<43} ║".format(start_time))
    print(Fore.CYAN + "║ Режим: {:<48} ║".format(mode))
    if config_path:
        display_path = config_path if len(config_path) <= 40 else "..." + config_path[-37:]
        print(Fore.CYAN + "║ Конфиг: {:<47} ║".format(display_path))
    print(Fore.CYAN + "╚" + "═" * 58 + "╝" + Style.RESET_ALL)


def print_help(topic: str = "main"):
    """Выводит справку по использованию."""
    helps = {
        "main": """
    СПРАВКА: Главное меню
    ----------------------
    1 - Режим пользователя (упрощённый доступ к основным функциям)
    2 - Режим администратора (полный доступ, требуется пароль)
    0 - Выход из программы

    В любом меню можно ввести '?' для вызова этой справки.
    """,
        "user": """
    СПРАВКА: Режим пользователя
    ---------------------------
    1 - Сгенерировать данные и обучить модель (подряд)
    2 - Только обучить модель (если данные уже есть)
    0 - Вернуться в главное меню
    """,
        "admin": """
    СПРАВКА: Режим администратора
    -----------------------------
    1  - Сгенерировать данные
    2  - Обучить модель (с детальными настройками)
    3  - Оценить модель на всех данных
    4  - Визуализировать данные
    5  - Показать статистику данных (расширенную)
    6  - Открыть папку с отчётами
    7  - Открыть папку с графиками
    8  - Изменить конфигурационный файл (интерактивно)
    9  - Просмотреть логи
    10 - Сравнить две модели
    11 - Построить карту дефектов
    12 - Запустить пользовательский скрипт
    13 - Показать информацию о системе
    14 - Анализ годографа дефекта (ёмкость/индуктивность)
    15 - Экспорт отчёта в HTML
    0  - Вернуться в главное меню
    """
    }
    print(Fore.MAGENTA + helps.get(topic, "Справка не найдена."))


def confirm_action(prompt: str, default: bool = False) -> bool:
    """Запрашивает подтверждение действия."""
    default_str = " (y/N)" if not default else " (Y/n)"
    answer = input(prompt + default_str + ": ").strip().lower()
    if default:
        return answer != 'n'
    else:
        return answer == 'y'


def run_script(script_name: str, config_path: str, extra_args: Optional[str] = None) -> bool:
    """Запускает указанный скрипт из папки scripts."""
    script_path = os.path.join(os.path.dirname(__file__), "scripts", script_name)
    if not os.path.exists(script_path):
        print(Fore.RED + f"Ошибка: скрипт {script_path} не найден.")
        return False

    cmd = [sys.executable, script_path, "--config", config_path]
    if extra_args:
        cmd.extend(extra_args.split())

    print(Fore.BLUE + f"Запуск: {' '.join(cmd)}")
    start = time.time()
    try:
        subprocess.run(cmd, check=True)
        elapsed = time.time() - start
        print(Fore.GREEN + f"Скрипт выполнен успешно за {elapsed:.2f} с.")
        log_action(f"Запущен скрипт {script_name} с конфигом {config_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(Fore.RED + f"Ошибка при выполнении скрипта: {e}")
        log_action(f"Ошибка при запуске {script_name}: {e}")
        return False


def open_folder(path: str) -> None:
    """Открывает указанную папку в системном файловом менеджере."""
    if not os.path.exists(path):
        print(Fore.RED + f"Папка {path} не существует.")
        return
    try:
        if sys.platform == 'win32':
            os.startfile(path)
        elif sys.platform == 'darwin':
            subprocess.run(['open', path])
        else:
            subprocess.run(['xdg-open', path])
        print(Fore.GREEN + f"Открыта папка: {path}")
        log_action(f"Открыта папка {path}")
    except Exception as e:
        print(Fore.RED + f"Не удалось открыть папку: {e}")


def show_data_stats(config_path: str, detailed: bool = False) -> None:
    """Выводит статистику по классам из файла данных. Если detailed=True, то также средние и дисперсии признаков."""
    try:
        config = load_config(config_path)
        data_path = config['paths']['data']
        if not os.path.exists(data_path):
            print(Fore.YELLOW + "Файл данных не найден. Сначала выполните генерацию.")
            return
        df = pd.read_csv(data_path)
        print(Fore.CYAN + "\n" + "="*60)
        print("СТАТИСТИКА ДАННЫХ")
        print("="*60)
        print(f"Всего записей: {len(df)}")
        print(f"Количество признаков: {len([c for c in df.columns if c not in ['class', 'x_position']])}")
        print("\nРаспределение по классам:")
        stats = df['class'].value_counts().sort_index()
        labels = ["Нет дефекта", "Утонение высоты", "Утонение ширины", "Утонение подложки", "Изменение εr"]
        for cls, count in stats.items():
            label = labels[cls] if cls < len(labels) else f"Класс {cls}"
            print(f"  {cls} ({label}): {count} образцов ({count/len(df)*100:.1f}%)")

        if detailed:
            # Статистика признаков по классам
            feature_cols = [c for c in df.columns if c not in ['class', 'x_position']]
            if len(feature_cols) > 0:
                print("\nСтатистика признаков (среднее ± стандартное отклонение) по классам:")
                for cls in sorted(df['class'].unique()):
                    subset = df[df['class'] == cls]
                    print(f"\nКласс {cls} ({labels[cls]}), n={len(subset)}")
                    # Выберем первые 5 признаков для примера, чтобы не засорять экран
                    for col in feature_cols[:5]:
                        mean = subset[col].mean()
                        std = subset[col].std()
                        print(f"  {col}: {mean:.4f} ± {std:.4f}")
                    if len(feature_cols) > 5:
                        print(f"  ... и ещё {len(feature_cols)-5} признаков")
        log_action("Просмотр статистики данных")
    except Exception as e:
        print(Fore.RED + f"Ошибка при чтении данных: {e}")


def train_and_show_results(config_path: str) -> None:
    """Загружает данные, обучает модель и выводит результаты."""
    try:
        config = load_config(config_path)
        data_path = config['paths']['data']
        if not os.path.exists(data_path):
            print(Fore.RED + "Файл данных не найден. Сначала выполните генерацию.")
            return

        df = pd.read_csv(data_path)
        feature_cols = [col for col in df.columns if col not in ['class', 'x_position']]
        X = df[feature_cols].values
        y = df['class'].values
        positions = df['x_position'].values

        start = time.time()
        result = train_and_evaluate(config, X, y, positions)
        elapsed = time.time() - start

        print(Fore.GREEN + "\n" + "="*60)
        print("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ")
        print("="*60)
        print(f"Accuracy на тесте: {result['accuracy']:.4f}")
        print(f"Macro F1: {result['macro_f1']:.4f}")
        print(f"Время обучения: {elapsed:.2f} с.")
        print("\nМетрики по классам:")
        print("Класс  Precision  Recall  F1     Support")
        for i, (p, r, f, s) in enumerate(zip(result['precision'], result['recall'], result['f1'], result['support'])):
            print(f"{i:5d}  {p:.4f}    {r:.4f}   {f:.4f}   {s:5d}")

        # Сохраняем отчёт
        report_path = os.path.join(config['paths']['reports'], 'classification_report.txt')
        save_report(result, config, feature_cols, report_path)
        print(Fore.GREEN + f"Отчёт сохранён в {report_path}")

        log_action(f"Обучение модели завершено, accuracy={result['accuracy']:.4f}")
    except Exception as e:
        print(Fore.RED + f"Ошибка при обучении: {e}")


def generate_and_train(config_path: str) -> None:
    """Последовательно генерирует данные и обучает модель."""
    if os.path.exists(load_config(config_path)['paths']['data']):
        if not confirm_action("Файл данных уже существует. Перезаписать?", default=False):
            print("Генерация отменена.")
            return
    print(Fore.CYAN + "Генерация данных...")
    try:
        config = load_config(config_path)
        ensure_dirs(config)
        start = time.time()
        df = generate_data(config)
        elapsed = time.time() - start
        print(Fore.GREEN + f"Данные успешно сгенерированы за {elapsed:.2f} с.: {df.shape[0]} строк.")
        log_action(f"Генерация данных: {df.shape[0]} строк")
        train_and_show_results(config_path)
    except Exception as e:
        print(Fore.RED + f"Ошибка при генерации или обучении: {e}")


def view_logs(config_path: str) -> None:
    """Просмотр последних записей лога."""
    try:
        config = load_config(config_path)
        log_file = config.get('logging', {}).get('file')
        if not log_file or not os.path.exists(log_file):
            print(Fore.YELLOW + "Лог-файл не найден или не указан в конфигурации.")
            return
        # Используем 'less' если доступен, иначе выводим последние строки
        if shutil.which('less'):
            subprocess.run(['less', log_file])
        else:
            with open(log_file, 'r') as f:
                lines = f.readlines()[-20:]  # последние 20 строк
            print(Fore.CYAN + "\nПоследние 20 строк лога:")
            for line in lines:
                print(line.rstrip())
        log_action("Просмотр логов")
    except Exception as e:
        print(Fore.RED + f"Ошибка при чтении лога: {e}")


def compare_models(config_path: str) -> None:
    """Сравнение двух сохранённых моделей."""
    model1 = input("Путь к первой модели (.pkl): ").strip()
    model2 = input("Путь ко второй модели (.pkl): ").strip()
    if not os.path.exists(model1) or not os.path.exists(model2):
        print(Fore.RED + "Один из файлов не найден.")
        return
    try:
        config = load_config(config_path)
        data_path = config['paths']['data']
        if not os.path.exists(data_path):
            print(Fore.RED + "Файл данных не найден.")
            return
        df = pd.read_csv(data_path)
        feature_cols = [col for col in df.columns if col not in ['class', 'x_position']]
        X = df[feature_cols].values
        y = df['class'].values

        model1_obj = joblib.load(model1)
        model2_obj = joblib.load(model2)
        y_pred1 = model1_obj.predict(X)
        y_pred2 = model2_obj.predict(X)

        acc1 = accuracy_score(y, y_pred1)
        acc2 = accuracy_score(y, y_pred2)
        print(Fore.CYAN + f"\nМодель 1 Accuracy: {acc1:.4f}")
        print(f"Модель 2 Accuracy: {acc2:.4f}")

        _, _, f1_1, _ = precision_recall_fscore_support(y, y_pred1, average=None)
        _, _, f1_2, _ = precision_recall_fscore_support(y, y_pred2, average=None)
        print("\nF1 по классам:")
        for i, (f1a, f1b) in enumerate(zip(f1_1, f1_2)):
            print(f"Класс {i}: {f1a:.4f} vs {f1b:.4f}")
        log_action("Сравнение моделей")
    except Exception as e:
        print(Fore.RED + f"Ошибка при сравнении: {e}")


def build_defect_map(config_path: str) -> None:
    """Строит карту дефектов на основе конфигурации."""
    try:
        config = load_config(config_path)
        line_params = config['line']
        def_cfg = config['defects']
        x_centers = def_cfg['positions']
        L_def = def_cfg['length']
        types = def_cfg['types']
        tp = def_cfg['type_params']

        parent = microstrip_line_t(
            W=line_params['width'],
            h=line_params['height'],
            t=line_params['thickness'],
            epsilon_r=line_params['epsilon_r']
        )

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

        colors = ['lightgray', 'red', 'green', 'blue', 'orange']
        labels = ['Нет дефекта', 'Утонение высоты', 'Утонение ширины',
                  'Утонение подложки', 'Изменение εr']

        save_path = os.path.join(config['paths']['figures'], 'defect_map.png')
        plot_defect_map(defects, types, line_params['length'], line_params['width'],
                        colors, labels, save_path=save_path)
        print(Fore.GREEN + f"Карта дефектов сохранена в {save_path}")
        log_action("Построена карта дефектов")
    except Exception as e:
        print(Fore.RED + f"Ошибка при построении карты: {e}")


def show_system_info() -> None:
    """Выводит информацию о системе и зависимостях."""
    print(Fore.CYAN + "\nИнформация о системе:")
    print(f"Платформа: {sys.platform}")
    print(f"Python: {sys.version}")
    print(f"Текущая директория: {os.getcwd()}")
    try:
        import numpy; print(f"numpy: {numpy.__version__}")
        import pandas; print(f"pandas: {pandas.__version__}")
        import matplotlib; print(f"matplotlib: {matplotlib.__version__}")
        import sklearn; print(f"scikit-learn: {sklearn.__version__}")
        import yaml; print(f"pyyaml: {yaml.__version__}")
        import joblib; print(f"joblib: {joblib.__version__}")
        import tqdm; print(f"tqdm: {tqdm.__version__}")
        import colorama; print(f"colorama: {colorama.__version__}")
    except ImportError as e:
        print(Fore.RED + f"Библиотека не установлена: {e}")
    log_action("Просмотр информации о системе")


def analyze_defect_interactive(config_path: str) -> None:
    """Интерактивный анализ дефекта по годографу."""
    try:
        config = load_config(config_path)
        data_path = config['paths']['data']
        if not os.path.exists(data_path):
            print(Fore.RED + "Файл данных не найден. Сначала выполните генерацию.")
            return
        df = pd.read_csv(data_path)
        
        freq_cols = sorted(set([col.split('_')[-1] for col in df.columns if col.startswith('I_Dx_')]))
        if not freq_cols:
            print(Fore.RED + "В данных нет столбцов I_Dx_*.")
            return
        
        print(Fore.CYAN + "\nДоступные частоты (ГГц):", ', '.join(freq_cols))
        
        print("Выберите режим анализа:")
        print(" 1. По классу дефекта")
        print(" 2. По координате x")
        mode = input("Ваш выбор: ").strip()
        
        if mode == '1':
            print("\nКлассы:")
            print("0 - Нет дефекта")
            print("1 - Утонение высоты")
            print("2 - Утонение ширины")
            print("3 - Утонение подложки")
            print("4 - Изменение εr")
            cls = input("Введите номер класса (0-4): ").strip()
            if not cls.isdigit() or int(cls) not in range(5):
                print(Fore.RED + "Неверный номер класса.")
                return
            result = analyze_from_dataframe(df, freq_cols, class_label=int(cls))
            print(Fore.GREEN + f"\nРезультат анализа для класса {cls}: {result}")
        elif mode == '2':
            x_pos = input("Введите координату x (в метрах): ").strip()
            try:
                x_pos = float(x_pos)
            except ValueError:
                print(Fore.RED + "Неверный формат числа.")
                return
            unique_pos = df['x_position'].unique()
            closest = min(unique_pos, key=lambda p: abs(p - x_pos))
            if abs(closest - x_pos) > 1e-6:
                print(Fore.YELLOW + f"Точной позиции нет, используем ближайшую: {closest}")
                x_pos = closest
            result = analyze_from_dataframe(df, freq_cols, x_position=x_pos)
            print(Fore.GREEN + f"\nРезультат анализа для позиции x = {x_pos}: {result}")
        else:
            print(Fore.RED + "Неверный выбор.")
            return
        
        if input("\nПостроить график фазы? (y/n): ").lower() == 'y':
            if mode == '1':
                subset = df[df['class'] == int(cls)]
            else:
                subset = df[np.isclose(df['x_position'], x_pos)]
            I_vals = []
            Q_vals = []
            for f in freq_cols:
                I_vals.append(subset[f'I_Dx_{f}'].values)
                Q_vals.append(subset[f'Q_Dx_{f}'].values)
            I_mean = np.mean(I_vals, axis=1)
            Q_mean = np.mean(Q_vals, axis=1)
            phase = np.angle(I_mean + 1j * Q_mean)
            phase_unwrapped = np.unwrap(phase)
            freqs_ghz = [float(f.replace('GHz', '')) for f in freq_cols]
            
            plt.figure(figsize=(8,5))
            plt.plot(freqs_ghz, np.degrees(phase_unwrapped), 'o-')
            plt.xlabel('Частота, ГГц')
            plt.ylabel('Фаза, градусы')
            plt.title('Зависимость фазы разностного канала Dx от частоты')
            plt.grid(True)
            plt.show()
        log_action(f"Анализ годографа: {result}")
    except Exception as e:
        print(Fore.RED + f"Ошибка при анализе: {e}")


def export_report_html(config_path: str) -> None:
    """Экспортирует отчёт классификации в HTML-формат."""
    try:
        config = load_config(config_path)
        report_txt = os.path.join(config['paths']['reports'], 'classification_report.txt')
        if not os.path.exists(report_txt):
            print(Fore.RED + "Сначала обучите модель, чтобы создать отчёт.")
            return
        with open(report_txt, 'r', encoding='utf-8') as f:
            content = f.read()
        html_content = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Отчёт классификации</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 40px; }}
h1 {{ color: #2c3e50; }}
pre {{ background: #f4f4f4; padding: 15px; border-radius: 5px; }}
</style>
</head>
<body>
<h1>Отчёт о классификации дефектов микрополосковой линии</h1>
<p>Сгенерировано: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
<pre>{content}</pre>
</body>
</html>"""
        html_path = os.path.join(config['paths']['reports'], 'classification_report.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(Fore.GREEN + f"HTML-отчёт сохранён в {html_path}")
        log_action("Экспорт отчёта в HTML")
    except Exception as e:
        print(Fore.RED + f"Ошибка при экспорте: {e}")


def edit_config_interactive(config_path: str) -> None:
    """Интерактивное редактирование конфигурационного файла."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(Fore.RED + f"Не удалось загрузить конфиг: {e}")
        return

    print(Fore.CYAN + "\nРедактирование конфигурации. Оставляйте поле пустым для сохранения текущего значения.")
    sections = {
        'line': 'Параметры линии',
        'defects': 'Параметры дефектов',
        'probe': 'Измерительная головка',
        'measurement': 'Параметры измерений',
        'scan': 'Параметры сканирования',
        'model': 'Параметры модели',
        'paths': 'Пути'
    }
    for sec, desc in sections.items():
        if sec not in config:
            continue
        print(Fore.YELLOW + f"\n--- {desc} ---")
        for key, value in config[sec].items():
            if isinstance(value, dict):
                # Вложенные словари, например frequencies или type_params
                print(Fore.MAGENTA + f"  {key}:")
                for subkey, subval in value.items():
                    new = input(f"    {subkey} [{subval}]: ").strip()
                    if new:
                        try:
                            # Попытка преобразовать в число, если это число
                            if isinstance(subval, (int, float)):
                                config[sec][key][subkey] = float(new) if '.' in new else int(new)
                            else:
                                config[sec][key][subkey] = new
                        except ValueError:
                            config[sec][key][subkey] = new
            else:
                new = input(f"  {key} [{value}]: ").strip()
                if new:
                    try:
                        if isinstance(value, bool):
                            config[sec][key] = new.lower() in ('true', 'yes', '1')
                        elif isinstance(value, (int, float)):
                            config[sec][key] = float(new) if '.' in new else int(new)
                        else:
                            config[sec][key] = new
                    except ValueError:
                        config[sec][key] = new

    # Сохраняем изменения
    backup_path = config_path + '.bak'
    shutil.copy2(config_path, backup_path)
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    print(Fore.GREEN + f"Конфигурация сохранена. Резервная копия: {backup_path}")
    log_action("Интерактивное редактирование конфигурации")


def main() -> None:
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config_path = os.path.join(os.path.dirname(__file__), "config", "default.yaml")

    if not os.path.exists(config_path):
        print(Fore.YELLOW + f"Предупреждение: файл конфигурации по умолчанию {config_path} не найден.")
        change = input("Указать другой путь? (y/n): ").strip().lower()
        if change == 'y':
            config_path = input("Введите путь к конфигурационному файлу: ").strip()
            if not os.path.exists(config_path):
                print(Fore.RED + "Файл не найден. Завершение.")
                return
        else:
            print(Fore.RED + "Работа без конфига невозможна. Завершение.")
            return

    while True:
        print_header(start_time, "ГЛАВНОЕ МЕНЮ", config_path)
        print(Fore.YELLOW + "Выберите режим:")
        print(" 1. Режим пользователя")
        print(" 2. Режим администратора")
        print(" 0. Выход")
        print(Fore.WHITE + " (Введите '?' для справки)" + Style.RESET_ALL)
        mode_choice = input("Ваш выбор: ").strip()

        if mode_choice == '?':
            print_help("main")
            input("Нажмите Enter, чтобы продолжить...")
            continue

        if mode_choice == '1':
            # Режим пользователя
            while True:
                print_header(start_time, "ПОЛЬЗОВАТЕЛЬ", config_path)
                print(Fore.GREEN + "\nРЕЖИМ ПОЛЬЗОВАТЕЛЯ")
                print(" 1. Сгенерировать данные и обучить модель (подряд)")
                print(" 2. Только обучить модель (если данные уже есть)")
                print(" 0. Вернуться в главное меню")
                print(Fore.WHITE + " (Введите '?' для справки)" + Style.RESET_ALL)
                choice = input("Выберите действие: ").strip()

                if choice == '?':
                    print_help("user")
                    input("Нажмите Enter, чтобы продолжить...")
                    continue

                if choice == '1':
                    generate_and_train(config_path)
                elif choice == '2':
                    train_and_show_results(config_path)
                elif choice == '0':
                    break
                else:
                    print(Fore.RED + "Неверный ввод.")
                input(Fore.WHITE + "\nНажмите Enter, чтобы продолжить..." + Style.RESET_ALL)

        elif mode_choice == '2':
            # Запрос пароля для входа в режим администратора
            attempts = 3
            authenticated = False
            while attempts > 0:
                pwd = getpass.getpass("Введите пароль администратора: ")
                if pwd == ADMIN_PASSWORD:
                    authenticated = True
                    break
                else:
                    attempts -= 1
                    print(Fore.RED + f"Неверный пароль. Осталось попыток: {attempts}")
            if not authenticated:
                print(Fore.RED + "Доступ запрещён.")
                continue

            # Режим администратора
            while True:
                print_header(start_time, "АДМИНИСТРАТОР", config_path)
                print(Fore.MAGENTA + "\nРЕЖИМ АДМИНИСТРАТОРА")
                print(" 1. Сгенерировать данные")
                print(" 2. Обучить модель (с детальными настройками)")
                print(" 3. Оценить модель на всех данных")
                print(" 4. Визуализировать данные")
                print(" 5. Показать статистику данных (расширенную)")
                print(" 6. Открыть папку с отчётами")
                print(" 7. Открыть папку с графиками")
                print(" 8. Изменить конфигурационный файл (интерактивно)")
                print(" 9. Просмотреть логи")
                print("10. Сравнить две модели")
                print("11. Построить карту дефектов")
                print("12. Запустить пользовательский скрипт")
                print("13. Показать информацию о системе")
                print("14. Анализ годографа дефекта (ёмкость/индуктивность)")
                print("15. Экспорт отчёта в HTML")
                print(" 0. Вернуться в главное меню")
                print(Fore.WHITE + " (Введите '?' для справки)" + Style.RESET_ALL)
                choice = input("Выберите действие: ").strip()

                if choice == '?':
                    print_help("admin")
                    input("Нажмите Enter, чтобы продолжить...")
                    continue

                if choice == '1':
                    if os.path.exists(load_config(config_path)['paths']['data']):
                        if confirm_action("Файл данных уже существует. Перезаписать?", default=False):
                            run_script("generate_data.py", config_path)
                    else:
                        run_script("generate_data.py", config_path)
                elif choice == '2':
                    train_and_show_results(config_path)
                elif choice == '3':
                    run_script("evaluate.py", config_path)
                elif choice == '4':
                    run_script("visualize_data.py", config_path)
                elif choice == '5':
                    show_data_stats(config_path, detailed=True)
                elif choice == '6':
                    try:
                        config = load_config(config_path)
                        reports_dir = config['paths']['reports']
                        if not os.path.isabs(reports_dir):
                            reports_dir = os.path.join(os.path.dirname(__file__), reports_dir)
                        open_folder(reports_dir)
                    except Exception as e:
                        print(Fore.RED + f"Ошибка: {e}")
                elif choice == '7':
                    try:
                        config = load_config(config_path)
                        figures_dir = config['paths']['figures']
                        if not os.path.isabs(figures_dir):
                            figures_dir = os.path.join(os.path.dirname(__file__), figures_dir)
                        open_folder(figures_dir)
                    except Exception as e:
                        print(Fore.RED + f"Ошибка: {e}")
                elif choice == '8':
                    edit_config_interactive(config_path)
                elif choice == '9':
                    view_logs(config_path)
                elif choice == '10':
                    compare_models(config_path)
                elif choice == '11':
                    build_defect_map(config_path)
                elif choice == '12':
                    script_name = input("Имя скрипта (например, my_script.py): ").strip()
                    extra = input("Дополнительные аргументы (если есть): ").strip()
                    run_script(script_name, config_path, extra if extra else None)
                elif choice == '13':
                    show_system_info()
                elif choice == '14':
                    analyze_defect_interactive(config_path)
                elif choice == '15':
                    export_report_html(config_path)
                elif choice == '0':
                    break
                else:
                    print(Fore.RED + "Неверный ввод.")
                input(Fore.WHITE + "\nНажмите Enter, чтобы продолжить..." + Style.RESET_ALL)

        elif mode_choice == '0':
            print(Fore.GREEN + "Выход.")
            log_action("Завершение программы")
            break
        else:
            print(Fore.RED + "Неверный выбор режима.")
            input("Нажмите Enter, чтобы продолжить...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(Fore.YELLOW + "\nПрограмма прервана пользователем.")
        log_action("Программа прервана пользователем")
        sys.exit(0)