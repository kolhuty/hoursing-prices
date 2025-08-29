# Housing Prices Prediction (Kaggle Dataset)

Этот проект посвящён предсказанию цен на жильё с использованием датасета House Prices - Advanced Regression Techniques
 с Kaggle.
Основная цель — построить и сравнить несколько моделей машинного обучения для решения задачи регрессии.

### Описание проекта

Exploratory Data Analysis (EDA)

Первичный анализ данных выполнен в Jupyter Notebook.

Были изучены распределения признаков, пропуски, выбросы и взаимосвязи между переменными.

Проведена обработка категориальных и числовых признаков.

Реализовано несколько моделей с помощью библиотеки scikit-learn:

**Линейные модели**:

- Baseline (Linear Regression)
- Ridge Regression
- Lasso Regression

**Деревья решений**:

- Decision Tree Regression 
- Random Forest Regression
- Gradient Boosting Tree Resression

**Подбор гиперпараметров**

Использовалась кросс-валидация (GridSearchCV) для оптимизации параметров моделей.

---

**Результаты**  
Метрика RMSLE на тесте Kaggle
- Lasso Regression: 0.227
- Gradient Boosting Tree: 0.127


| Model            | Best Parameters / Alpha                                      | CV RMSE |
|------------------|---------------------------------------------------------------|---------|
| Baseline CV      | -                                                             | 55,591$ |
| Ridge            | alpha=3.0                                                    | 55,344$ |
| Lasso            | alpha=0.1                                                    | 49,293$ |
| DecisionTree     | max_depth=10, min_samples_leaf=8                             | 39,352$ |
| RandomForest     | max_depth=None, min_samples_leaf=1, n_estimators=300         | 30,754$ |
| GradientBoosting | learning_rate=0.1, max_depth=3, n_estimators=300            | 29,284$ |
| XGBoost          | learning_rate=0.05, max_depth=3, n_estimators=500           | 29,541$ |

Наилучший результат показала модель Gradient Boosting  
Среди линейных моделей лучше всего сработала Lasso Regression

Датасет взят с Kaggle: https://www.kaggle.com/c/house-prices-advanced-regression-techniques