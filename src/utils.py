import numpy
import datetime
import os
import yaml


def load_config(config_path):
    """
    Загружает конфигурацию из YAML-файла.

    Параметры
    ---------
    config_path : str
        Путь к файлу конфигурации.

    Возвращает
    ----------
    dict
        Словарь с параметрами конфигурации.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_report(result, config, feature_names, path):
    """
    Формирует текстовый отчёт о классификации и сохраняет его в файл.

    Параметры
    ---------
    result : dict
        Словарь с результатами обучения (вывод train_and_evaluate).
    config : dict
        Словарь конфигурации (для отображения параметров).
    feature_names : list
        Список названий признаков.
    path : str
        Путь для сохранения отчёта (включая имя файла).
    """
    lines = []
    lines.append("=" * 60)
    lines.append("ОТЧЁТ О КЛАССИФИКАЦИИ ДЕФЕКТОВ МИКРОПОЛОСКОВОЙ ЛИНИИ")
    lines.append(f"Дата и время: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)

    # Параметры модели
    lines.append("\n1. ПАРАМЕТРЫ МОДЕЛИ")
    lines.append(f"   Лучшие гиперпараметры (GridSearchCV):")
    best_params = result.get('best_params', {})
    for k, v in best_params.items():
        lines.append(f"      {k}: {v}")

    # Размеры выборок (можно взять из конфига, но точные размеры не сохраняются в result)
    # Поэтому просто выведем информацию из конфига
    lines.append(f"\n   Размер обучающей выборки: {config['model'].get('train_size', 'не указано')}")
    lines.append(f"   Размер тестовой выборки: {config['model'].get('test_size', 'не указано')}")

    # Метрики
    lines.append("\n2. МЕТРИКИ КЛАССИФИКАЦИИ")
    lines.append(f"   Accuracy на тесте: {result.get('accuracy', 0):.4f}")
    lines.append(f"   Macro F1 на тесте: {result.get('macro_f1', 0):.4f}\n")
    lines.append("   Метрики по классам:")
    lines.append("   Класс  Precision  Recall  F1     Support")

    precision = result.get('precision', [])
    recall = result.get('recall', [])
    f1 = result.get('f1', [])
    support = result.get('support', [])

    for i in range(len(precision)):
        lines.append(f"   {i:5d}  {precision[i]:.4f}    {recall[i]:.4f}   {f1[i]:.4f}   {support[i]:5d}")

    # Важность признаков (топ-10)
    importances = result.get('feature_importances', [])
    if len(importances) > 0:
        indices = numpy.argsort(importances)[::-1][:10]
        top_features = [(feature_names[i], importances[i]) for i in indices]

        lines.append("\n3. ВАЖНОСТЬ ПРИЗНАКОВ (ТОП-10)")
        for i, (name, imp) in enumerate(top_features, 1):
            lines.append(f"   {i:2d}. {name:30s} : {imp:.4f}")

    # Матрица ошибок
    cm = result.get('confusion_matrix')
    if cm is not None:
        lines.append("\n4. МАТРИЦА ОШИБОК")
        # Преобразуем матрицу в строку с выравниванием
        cm_str = numpy.array2string(cm, separator=', ', formatter={'int': lambda x: f'{x:3d}'})
        lines.append(cm_str)

    lines.append("\n" + "=" * 60)
    lines.append("КОНЕЦ ОТЧЁТА")
    lines.append("=" * 60)

    report_text = "\n".join(lines)

    # Запись в файл
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"Отчёт сохранён в {path}")


def ensure_dirs(config):
    """
    Создаёт директории, указанные в конфигурации (для данных, моделей, отчётов),
    если они не существуют.

    Параметры
    ---------
    config : dict
        Словарь конфигурации, содержащий раздел 'paths'.
    """
    paths = config.get('paths', {})
    for key, path in paths.items():
        if isinstance(path, str):
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Создана директория: {directory}")