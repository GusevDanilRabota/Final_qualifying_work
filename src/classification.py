import numpy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import joblib


def train_and_evaluate(config, X, y, positions=None):
    """
    Обучает классификатор случайного леса на данных X, y.
    Выполняет поиск гиперпараметров по сетке на обучающей выборке,
    затем обучает финальную модель на объединённых train+val и
    оценивает на тестовой выборке.

    Параметры
    ---------
    config : dict
        Словарь конфигурации (разделы model, paths).
    X : numpy.ndarray
        Массив признаков формы (n_samples, n_features).
    y : numpy.ndarray
        Вектор меток классов.
    positions : numpy.ndarray, optional
        Координаты x для каждой строки (если нужны для визуализации).

    Возвращает
    ----------
    dict
        Словарь с результатами:
            - best_params : dict
            - accuracy : float
            - macro_f1 : float
            - precision : numpy.ndarray
            - recall : numpy.ndarray
            - f1 : numpy.ndarray
            - support : numpy.ndarray
            - confusion_matrix : numpy.ndarray
            - y_test : numpy.ndarray
            - y_pred : numpy.ndarray
            - feature_importances : numpy.ndarray
            - model : RandomForestClassifier
            - pos_test : numpy.ndarray (если positions заданы)
    """
    # Извлекаем параметры из конфига
    test_size = config['model']['test_size']
    val_size_from_temp = config['model']['val_size_from_temp']
    random_state = config['model']['random_state']
    rf_params_grid = config['model']['rf_params']
    cv_folds = config['model']['cv_folds']

    # Первое разделение: обучающая (70%) и временная (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Второе разделение: временная на валидационную (15%) и тестовую (15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=val_size_from_temp,
        random_state=random_state,
        stratify=y_temp
    )

    # --- Поиск гиперпараметров на обучающей выборке ---
    rf_base = RandomForestClassifier(
        random_state=random_state,
        n_jobs=-1
    )

    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state
    )

    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=rf_params_grid,
        cv=cv,
        scoring='f1_macro',
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_

    # --- Обучение финальной модели на train + val ---
    X_train_full = numpy.vstack([X_train, X_val])
    y_train_full = numpy.hstack([y_train, y_val])

    best_rf = RandomForestClassifier(
        **best_params,
        random_state=random_state,
        n_jobs=-1
    )
    best_rf.fit(X_train_full, y_train_full)

    # --- Оценка на тестовой выборке ---
    y_pred = best_rf.predict(X_test)
    accuracy = numpy.mean(y_pred == y_test)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred
    )
    macro_f1 = numpy.mean(f1)

    # Сохраняем модель
    joblib.dump(best_rf, config['paths']['model'])

    # Формируем результат
    result = {
        'best_params': best_params,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'y_test': y_test,
        'y_pred': y_pred,
        'feature_importances': best_rf.feature_importances_,
        'model': best_rf
    }

    # Если переданы координаты, сохраняем их для тестовой выборки
    if positions is not None:
        # Повторяем то же разделение для позиций
        _, _, _, _, pos_train, pos_temp = train_test_split(
            X, y, positions,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        _, _, _, _, pos_val, pos_test = train_test_split(
            X_temp, y_temp, pos_temp,
            test_size=val_size_from_temp,
            random_state=random_state,
            stratify=y_temp
        )
        result['pos_test'] = pos_test

    return result