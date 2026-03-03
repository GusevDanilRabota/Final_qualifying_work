# visualization.py
import numpy
import matplotlib.pyplot
from typing import List, Optional, Any
from matplotlib.patches import Rectangle, Patch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.tree import plot_tree

# Для type hints
import pandas


def plot_defect_map(
    defects: List[Any],
    types: List[int],
    L_line: float,
    W_nom: float,
    colors: List[str],
    labels: List[str],
    title: str = "Карта дефектов на микрополосковой линии",
    save_path: Optional[str] = None
) -> None:
    """
    Рисует карту дефектов: прямоугольники дефектов с подписями типов.
    """
    fig, ax = matplotlib.pyplot.subplots(figsize=(12, 3))
    ax.set_xlim(0, L_line * 1000)
    ax.set_ylim(0, W_nom * 1000)
    ax.set_xlabel('x, мм')
    ax.set_ylabel('y, мм')
    ax.set_title(title)

    # Сетка 1 мм
    for x in numpy.arange(0, L_line * 1000 + 1, 1):
        ax.axvline(x, color='lightgray', linewidth=0.5, linestyle='-', alpha=0.3)
    for y in numpy.arange(0, W_nom * 1000 + 1, 1):
        ax.axhline(y, color='lightgray', linewidth=0.5, linestyle='-', alpha=0.3)

    # Прямоугольники дефектов
    for defect, typ in zip(defects, types):
        x_left = defect.x1 * 1000
        width = defect.L_def * 1000
        rect = Rectangle((x_left, 0), width, W_nom * 1000,
                         linewidth=2, edgecolor='black',
                         facecolor=colors[typ], alpha=0.5)
        ax.add_patch(rect)
        ax.text(x_left + width / 2, W_nom * 1000 / 2, str(typ),
                ha='center', va='center', fontsize=12,
                color='black', weight='bold')

    # Легенда
    legend_elements = [Patch(facecolor=colors[i], alpha=0.5,
                             edgecolor='black', label=labels[i])
                       for i in range(len(colors))]
    ax.legend(handles=legend_elements, loc='upper right')

    matplotlib.pyplot.tight_layout()
    if save_path:
        matplotlib.pyplot.savefig(save_path, dpi=150)
    matplotlib.pyplot.show()


def plot_hodographs(
    df: pandas.DataFrame,
    freq: float,
    classes: List[int],
    colors: List[str],
    labels: List[str],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    random_state: int = 42
) -> None:
    """
    Годографы разностного канала Dx на указанной частоте.
    """
    col_I = f'I_Dx_{freq}GHz'
    col_Q = f'Q_Dx_{freq}GHz'

    matplotlib.pyplot.figure(figsize=(8, 6))

    for cls, color, label in zip(classes, colors[:len(classes)], labels):
        subset = df[df['class'] == cls]
        if len(subset) > 50:
            subset = subset.sample(50, random_state=random_state)
        I = subset[col_I].values
        Q = subset[col_Q].values
        matplotlib.pyplot.scatter(I, Q, color=color, alpha=0.6, label=label, s=30)

    matplotlib.pyplot.xlabel('I (синфазная), В')
    matplotlib.pyplot.ylabel('Q (квадратурная), В')
    if title is None:
        title = f'Годографы разностного канала Dx на частоте {freq} ГГц'
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.grid(True, linestyle='--', alpha=0.5)
    matplotlib.pyplot.legend()
    matplotlib.pyplot.axis('equal')
    matplotlib.pyplot.tight_layout()
    if save_path:
        matplotlib.pyplot.savefig(save_path, dpi=150)
    matplotlib.pyplot.show()


def plot_per_frequency_scatter(
    df: pandas.DataFrame,
    freqs: List[float],
    classes: List[int],
    colors: List[str],
    labels: List[str],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    random_state: int = 42
) -> None:
    """
    Многопанельный график рассеяния I-Q для всех частот (канал Dx).
    """
    n_freqs = len(freqs)
    fig, axes = matplotlib.pyplot.subplots(2, (n_freqs + 1) // 2, figsize=(20, 8))
    axes = axes.flatten()

    # Вычисляем общие пределы для всех подграфиков
    all_I = []
    all_Q = []
    for f in freqs:
        col_I = f'I_Dx_{f}GHz'
        col_Q = f'Q_Dx_{f}GHz'
        all_I.extend(df[col_I].values)
        all_Q.extend(df[col_Q].values)
    I_min, I_max = numpy.min(all_I), numpy.max(all_I)
    Q_min, Q_max = numpy.min(all_Q), numpy.max(all_Q)
    margin = 0.1 * max(I_max - I_min, Q_max - Q_min)

    for i, f in enumerate(freqs):
        col_I = f'I_Dx_{f}GHz'
        col_Q = f'Q_Dx_{f}GHz'
        ax = axes[i]
        has_data = False
        for cls, color, label in zip(classes, colors[:len(classes)], labels):
            subset = df[df['class'] == cls]
            if len(subset) == 0:
                continue
            if len(subset) > 200:
                subset = subset.sample(200, random_state=random_state)
            I = subset[col_I].values
            Q = subset[col_Q].values
            ax.scatter(I, Q, color=color, alpha=0.4, s=10,
                       label=label if i == 0 else "")
            has_data = True
        ax.set_title(f'{f} ГГц')
        ax.set_xlabel('I')
        ax.set_ylabel('Q')
        ax.grid(True, linestyle='--', alpha=0.3)
        if has_data:
            ax.set_xlim(I_min - margin, I_max + margin)
            ax.set_ylim(Q_min - margin, Q_max + margin)

    # Убираем лишние подграфики
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Общая легенда
    handles = [matplotlib.pyplot.Line2D([0], [0], marker='o', color='w',
                                        markerfacecolor=colors[i], markersize=8,
                                        label=labels[i])
               for i in range(len(classes)) if len(df[df['class'] == i]) > 0]
    fig.legend(handles=handles, loc='upper center',
               bbox_to_anchor=(0.5, 1.02), ncol=5, fontsize=12)

    if title is None:
        title = 'Разделение классов на плоскости I-Q разностного канала Dx'
    matplotlib.pyplot.suptitle(title, y=1.05, fontsize=14)
    matplotlib.pyplot.tight_layout()
    if save_path:
        matplotlib.pyplot.savefig(save_path, dpi=150, bbox_inches='tight')
    matplotlib.pyplot.show()


def plot_pca(
    X: numpy.ndarray,
    y: numpy.ndarray,
    classes: List[int],
    colors: List[str],
    labels: List[str],
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> tuple:
    """
    Проекция всех признаков на первые две главные компоненты.
    """
    # Стандартизация
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_.sum()
    print(f"Доля объяснённой дисперсии двумя компонентами: {explained:.3f}")

    matplotlib.pyplot.figure(figsize=(10, 8))
    for cls, color, label in zip(classes, colors[:len(classes)], labels):
        mask = (y == cls)
        matplotlib.pyplot.scatter(X_pca[mask, 0], X_pca[mask, 1],
                                   color=color, alpha=0.5, s=20, label=label)

    matplotlib.pyplot.xlabel('PC1')
    matplotlib.pyplot.ylabel('PC2')
    if title is None:
        title = 'Проекция всех признаков на первые две главные компоненты'
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.grid(True, linestyle='--', alpha=0.3)
    matplotlib.pyplot.legend()
    matplotlib.pyplot.tight_layout()
    if save_path:
        matplotlib.pyplot.savefig(save_path, dpi=150)
    matplotlib.pyplot.show()

    return pca, X_pca


def plot_frequency_dependence(
    df: pandas.DataFrame,
    freqs: List[float],
    classes: List[int],
    colors: List[str],
    labels: List[str],
    channel: str = 'I_S',
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Частотная зависимость среднего значения указанного канала.
    """
    mean_vals = {cls: [] for cls in classes}
    std_vals = {cls: [] for cls in classes}

    for f in freqs:
        col = f'{channel}_{f}GHz'
        for cls in classes:
            values = df[df['class'] == cls][col].values
            mean_vals[cls].append(numpy.mean(values))
            std_vals[cls].append(numpy.std(values))

    matplotlib.pyplot.figure(figsize=(10, 6))
    for cls, color, label in zip(classes, colors[:len(classes)], labels):
        matplotlib.pyplot.errorbar(freqs, mean_vals[cls], yerr=std_vals[cls],
                                    color=color, marker='o', capsize=3,
                                    label=label, alpha=0.7)

    matplotlib.pyplot.xlabel('Частота, ГГц')
    matplotlib.pyplot.ylabel(f'Среднее значение {channel}, В')
    if title is None:
        title = f'Частотная зависимость {channel}'
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.grid(True, linestyle='--', alpha=0.3)
    matplotlib.pyplot.legend()
    matplotlib.pyplot.tight_layout()
    if save_path:
        matplotlib.pyplot.savefig(save_path, dpi=150)
    matplotlib.pyplot.show()


def plot_phase_frequency(
    df: pandas.DataFrame,
    freqs: List[float],
    classes: List[int],
    colors: List[str],
    labels: List[str],
    channel: str = 'Dx',
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Частотная зависимость фазы комплексного сигнала разностного канала.
    """
    matplotlib.pyplot.figure(figsize=(10, 6))

    for cls, color, label in zip(classes, colors[:len(classes)], labels):
        phase_means = []
        for f in freqs:
            I = df[df['class'] == cls][f'I_{channel}_{f}GHz'].values
            Q = df[df['class'] == cls][f'Q_{channel}_{f}GHz'].values
            phase = numpy.angle(I + 1j * Q)  # радианы
            phase_means.append(numpy.mean(phase))
        matplotlib.pyplot.plot(freqs, numpy.rad2deg(phase_means),
                                marker='o', color=color, label=label)

    matplotlib.pyplot.xlabel('Частота, ГГц')
    matplotlib.pyplot.ylabel('Средняя фаза, градусы')
    if title is None:
        title = f'Частотная зависимость фазы разностного канала {channel}'
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.grid(True, linestyle='--', alpha=0.3)
    matplotlib.pyplot.legend()
    matplotlib.pyplot.tight_layout()
    if save_path:
        matplotlib.pyplot.savefig(save_path, dpi=150)
    matplotlib.pyplot.show()


def plot_feature_importance(
    importances: numpy.ndarray,
    feature_names: List[str],
    top_n: int = 20,
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Столбчатая диаграмма важности признаков.
    """
    indices = numpy.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in indices]
    top_imps = [importances[i] for i in indices]

    matplotlib.pyplot.figure(figsize=(10, 8))
    matplotlib.pyplot.barh(range(len(top_names)), top_imps[::-1], align='center')
    matplotlib.pyplot.yticks(range(len(top_names)), top_names[::-1])
    matplotlib.pyplot.xlabel('Важность')
    if title is None:
        title = f'Топ-{top_n} важных признаков'
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.tight_layout()
    if save_path:
        matplotlib.pyplot.savefig(save_path, dpi=150)
    matplotlib.pyplot.show()


def plot_tree_fragment(
    model,
    feature_names: List[str],
    class_names: List[str],
    max_depth: int = 3,
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Визуализация фрагмента первого дерева из случайного леса.
    """
    first_tree = model.estimators_[0]
    matplotlib.pyplot.figure(figsize=(20, 10))
    plot_tree(first_tree, max_depth=max_depth,
              feature_names=feature_names,
              class_names=class_names,
              filled=True, rounded=True, fontsize=10)
    if title is None:
        title = f'Фрагмент первого дерева (первые {max_depth} уровня)'
    matplotlib.pyplot.title(title)
    if save_path:
        matplotlib.pyplot.savefig(save_path, dpi=150, bbox_inches='tight')
    matplotlib.pyplot.show()


def plot_confusion_matrix(
    y_test: numpy.ndarray,
    y_pred: numpy.ndarray,
    classes: List[int],
    title: str = 'Матрица ошибок',
    save_path: Optional[str] = None
) -> None:
    """
    Отображает матрицу ошибок.
    """
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=classes)
    disp.plot(cmap='Blues', values_format='d')
    matplotlib.pyplot.title(title)
    if save_path:
        matplotlib.pyplot.savefig(save_path, dpi=150)
    matplotlib.pyplot.show()


def plot_comparison_map(
    y_test: numpy.ndarray,
    y_pred: numpy.ndarray,
    pos_test: numpy.ndarray,
    defects: List[Any],
    types: List[int],
    L_line: float,
    W_nom: float,
    colors: List[str],
    labels: List[str],
    save_path: Optional[str] = None
) -> None:
    """
    Сравнение истинной и предсказанной карты дефектов для тестовых позиций.
    """
    fig, (ax1, ax2) = matplotlib.pyplot.subplots(2, 1, figsize=(12, 6), sharex=True)

    for ax in (ax1, ax2):
        ax.set_xlim(0, L_line * 1000)
        ax.set_ylim(0, W_nom * 1000)
        ax.set_ylabel('y, мм')
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)

    ax2.set_xlabel('x, мм')
    ax1.set_title('Истинная карта дефектов с тестовыми позициями')
    ax2.set_title('Предсказанная карта дефектов')

    # Рисуем прямоугольники дефектов на обоих графиках
    for defect, typ in zip(defects, types):
        x_left = defect.x1 * 1000
        width = defect.L_def * 1000
        rect1 = Rectangle((x_left, 0), width, W_nom * 1000,
                          linewidth=1, edgecolor='black',
                          facecolor=colors[typ], alpha=0.3)
        ax1.add_patch(rect1)
        rect2 = Rectangle((x_left, 0), width, W_nom * 1000,
                          linewidth=1, edgecolor='black',
                          facecolor=colors[typ], alpha=0.3)
        ax2.add_patch(rect2)

    # Добавляем сетку 1 мм
    for x in numpy.arange(0, L_line * 1000 + 1, 1):
        ax1.axvline(x, color='lightgray', linewidth=0.3, alpha=0.2)
        ax2.axvline(x, color='lightgray', linewidth=0.3, alpha=0.2)
    for y in numpy.arange(0, W_nom * 1000 + 1, 1):
        ax1.axhline(y, color='lightgray', linewidth=0.3, alpha=0.2)

    # Истинные классы
    for cls in range(len(colors)):
        mask = (y_test == cls)
        ax1.scatter(pos_test[mask] * 1000,
                    [W_nom * 1000 / 2] * numpy.sum(mask),
                    color=colors[cls], s=30, edgecolor='black',
                    linewidth=0.5, label=labels[cls] if cls == 0 else "")

    # Предсказанные классы
    for cls in range(len(colors)):
        mask = (y_pred == cls)
        ax2.scatter(pos_test[mask] * 1000,
                    [W_nom * 1000 / 2] * numpy.sum(mask),
                    color=colors[cls], s=30, edgecolor='black',
                    linewidth=0.5, label=labels[cls] if cls == 0 else "")

    # Легенда общая
    handles = [Patch(facecolor=colors[i], edgecolor='black',
                     alpha=0.7, label=labels[i])
               for i in range(len(colors))]
    fig.legend(handles=handles, loc='upper center',
               bbox_to_anchor=(0.5, 1.05), ncol=5, fontsize=10)

    matplotlib.pyplot.tight_layout()
    if save_path:
        matplotlib.pyplot.savefig(save_path, dpi=150, bbox_inches='tight')
    matplotlib.pyplot.show()