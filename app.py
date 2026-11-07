from pathlib import Path
import json
from io import StringIO

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

BASE = Path(__file__).resolve().parent


def signature(path):
    info = path.stat()
    return (info.st_mtime_ns, info.st_size)


@st.cache_resource
def load_model(model_signature):
    # Сигнатура меняет ключ при замене модели.
    return joblib.load(BASE / 'model.joblib')


@st.cache_data
def load_test(test_signature):
    return pd.read_csv(BASE / 'test.csv')


@st.cache_data
def test_probabilities(_model, X, model_signature):
    # Порог не входит в ключ: при сдвиге слайдера вероятности берутся из кэша.
    return _model.predict_proba(X)[:, 1]


def read_batch(content, columns):
    """Проверяем ввод до передачи модели, сохраняем исходные единицы."""
    try:
        frame = pd.read_csv(StringIO(content.decode('utf-8-sig')))
    except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise ValueError('Не удалось прочитать CSV в UTF-8 с разделителем запятая.') from error
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError('Не хватает обязательных столбцов: ' + ', '.join(missing))
    if frame.empty:
        raise ValueError('Файл не содержит строк данных.')
    try:
        features = frame.loc[:, columns].apply(pd.to_numeric, errors='raise')
    except (ValueError, TypeError) as error:
        raise ValueError('Все признаки должны содержать числа.') from error
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError('В признаках есть пропуски или бесконечные значения.')
    if not features['is_red'].isin([0, 1]).all():
        raise ValueError('is_red должен быть 0 (белое) или 1 (красное).')
    return features


def batch_result(model, features, threshold):
    result = features.copy()
    result['probability_good'] = model.predict_proba(features)[:, 1]
    result['prediction'] = (result['probability_good'] >= threshold).astype(int)
    return result


def main():
    st.set_page_config(page_title='Рекомендация вин', layout='wide')
    config = json.loads((BASE / 'config.json').read_text())
    columns = config['feature_columns']
    model_signature = signature(BASE / 'model.joblib')
    model = load_model(model_signature)
    test = load_test(signature(BASE / 'test.csv'))
    X_test = test.loc[:, columns]
    y_test = test['target']
    p_test = test_probabilities(model, X_test, model_signature)

    st.title('Рекомендация вин')
    st.caption('Хорошее вино: исходная экспертная оценка не ниже 7.')
    threshold = st.sidebar.slider('Порог рекомендации', 0.0, 1.0,
                                  value=float(config['threshold']), step=0.001,
                                  format='%.6f', key='threshold')
    st.sidebar.caption(f"Порог из config.json: {config['threshold']:.12f}")
    single, scores, batch, weights = st.tabs(
        ['Одно вино', 'Модельные метрики', 'Пакетный прогноз', 'Коэффициенты'])

    with single:
        values = {}
        left, right = st.columns(2)
        for i, column in enumerate(columns):
            with (left if i % 2 == 0 else right):
                if column == 'is_red':
                    values[column] = int(st.radio('Цвет вина', ['белое', 'красное']) == 'красное')
                else:
                    values[column] = st.number_input(
                        column, min_value=float(X_test[column].min()),
                        max_value=float(X_test[column].max()),
                        value=float(X_test[column].median()), format='%.5f')
        wine = pd.DataFrame([values], columns=columns)
        probability = float(model.predict_proba(wine)[0, 1])
        st.metric('Вероятность хорошего вина', f'{probability:.2%}')
        st.write('Рекомендовать' if probability >= threshold else 'Не рекомендовать')

    with scores:
        pred = (p_test >= threshold).astype(int)
        cm = metrics.confusion_matrix(y_test, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        precision = metrics.precision_score(y_test, pred, zero_division=0)
        recall = metrics.recall_score(y_test, pred, zero_division=0)
        f1 = metrics.f1_score(y_test, pred, zero_division=0)
        for column, label, value in zip(st.columns(3), ['Precision', 'Recall', 'F1'], [precision, recall, f1]):
            column.metric(label, f'{value:.4f}')
        st.write(f'Рекомендаций: {int(pred.sum())} из {len(pred)}')
        fig, ax = plt.subplots()
        metrics.ConfusionMatrixDisplay(cm, display_labels=['Остальные', 'Хорошие']).plot(ax=ax, colorbar=False)
        st.pyplot(fig)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        fpr, tpr, _ = metrics.roc_curve(y_test, p_test)
        axes[0].plot(fpr, tpr)
        axes[0].scatter(fp / (fp + tn) if fp + tn else 0, recall, color='red', label='Текущий порог')
        axes[0].set(xlabel='FPR', ylabel='Recall', title='ROC')
        pr, re, _ = metrics.precision_recall_curve(y_test, p_test)
        axes[1].plot(re, pr)
        axes[1].axhline(y_test.mean(), linestyle='--', label='Случайный ориентир')
        axes[1].scatter(recall, precision, color='red', label='Текущий порог')
        axes[1].set(xlabel='Recall', ylabel='Precision', title='PR')
        for ax in axes:
            ax.legend()
        st.pyplot(fig)
        plt.close(fig)
        st.write('Повышение порога уменьшает или сохраняет число рекомендаций и recall. '
                 'Снижение расширяет подборку и может добавить посредственные вина. '
                 'Precision не обязана меняться монотонно. Кривые от порога не зависят.')
        st.caption('В исходных данных есть повторы между train и test; оценка на новых образцах может быть ниже.')

    with batch:
        st.write('Загрузите CSV в UTF-8 с 12 признаками в исходных единицах, без target.')
        uploaded = st.file_uploader('CSV с винами', type=['csv'])
        if uploaded is not None:
            try:
                features = read_batch(uploaded.getvalue(), columns)
                result = batch_result(model, features, threshold)
            except ValueError as error:
                st.error(str(error))
            else:
                st.dataframe(result)
                st.download_button('Скачать прогноз', result.to_csv(index=False).encode('utf-8'),
                                   file_name='wine_predictions.csv', mime='text/csv')

    with weights:
        coefficients = pd.Series(model.steps[-1][1].coef_.ravel(), index=columns).sort_values()
        fig, ax = plt.subplots(figsize=(9, 5))
        coefficients.plot.barh(ax=ax)
        ax.set_xlabel('Коэффициент после масштабирования')
        st.pyplot(fig)
        plt.close(fig)
        st.write(f'Самый сильный положительный коэффициент: {coefficients.idxmax()}; '
                 f'отрицательный: {coefficients.idxmin()}.')
        st.caption('Коэффициенты описывают условные связи признаков с прогнозом. '
                   'Они не доказывают причинного влияния на качество; признаки могут быть связаны между собой.')


if __name__ == '__main__':
    main()
