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
Метрика RMSE на тесте Kaggle
- Lasso Regression: 0.227
- Gradient Boosting Tree: 0.127

Наилучший результат показала модель Gradient Boosting  
Среди линейных моделей лучше всего сработала Lasso Regression