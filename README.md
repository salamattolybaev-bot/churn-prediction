# Customer Churn Prediction (Telecom)
## Project Overview

Проект посвящён прогнозированию оттока клиентов телеком-компании на основе поведенческих, тарифных и сервисных данных.
Основная цель — построить устойчивую ML-модель, оптимизированную под бизнес-метрики, и корректно сравнить несколько алгоритмов.

## Dataset

Источник: Churn Prediction 2025 (Kaggle)

Целевая переменная: churn (1 — клиент ушёл, 0 — остался)

Дисбаланс классов: ~14% churn

## Data Preparation & EDA

В ходе EDA были выявлены проблемы качества данных, характерные для реальных бизнес-датасетов:

- часть числовых признаков представлена строками;

- обнаружены системные ошибки форматирования (, вместо ., символы o / l вместо 0 / 1).

### Принятые решения:

- нормализация и приведение признаков к числовому типу;

- пересчёт агрегированного признака Total charge как суммы компонент
(day / eve / night / intl) для согласованности данных.

Это позволило устранить шум, повысить стабильность моделей и интерпретируемость признаков.

## Feature Engineering

- Target Encoding для признака state (без data leakage, внутри CV)

- One-Hot Encoding для plans, sex

- Масштабирование числовых признаков (для линейных моделей)

- Подбор оптимального порога классификации по F1-score

## Models

Обучены и сравнены модели:

- Logistic Regression (baseline)

- Random Forest

- Gradient Boosting

- XGBoost

- LightGBM

- CatBoost


Для всех моделей:

- использован StratifiedKFold (5 folds)

- метрики усреднены по фолдам

- учтён дисбаланс классов (scale_pos_weight / class weights)

## Cross-Validation Results

| model     | roc_auc  | best_threshold | precision | recall   | f1       |
|-----------|----------|----------------|-----------|----------|----------|
| CatBoost  | 0.889    | 0.649          | 0.902     | 0.698    | 0.785    |
| GB        | 0.888    | 0.457          | 0.981     | 0.717    | 0.828    |
| RF        | 0.883    | 0.402          | 0.957     | 0.726    | 0.824    |
| XGBoost   | 0.878    | 0.631          | 0.980     | 0.715    | 0.826    |
| LightGBM  | 0.867    | 0.811          | 0.992     | 0.718    | 0.831    |
| LogReg    | 0.722    | 0.217          | 0.358     | 0.519    | 0.414    |

## Business Impact Analysis

Предположим:

- 1 000 000 клиентов

- churn rate ≈ 14% (140 000 клиентов)

- средний LTV = $40

### LightGBM (Recall ≈ 0.72):

- выявляется ≈ 100 000 клиентов, склонных к уходу

- при удержании хотя бы 30%:

   - ≈ 30 000 клиентов сохранено

   - ≈ $1 200 000 сохранённой выручки(на каждые 1 млн клиентов)

Модель напрямую поддерживает retention-стратегии и ROI-ориентированные решения.

## Production Considerations

- Monitoring: ROC AUC, Recall@Threshold, churn rate drift

- Data drift: контроль распределений (PSI, feature means)

- Retraining: раз в 1–3 месяца или при деградации метрик

- Threshold tuning: адаптация под бизнес-стоимость ошибок

## Key Takeaways

- Работа с грязными данными, а не «идеальным» датасетом

- Корректный CV-подход без leakage

- Оптимизация под бизнес-метрики, а не accuracy

- Сравнение моделей на едином фреймворке

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, CatBoost, Matplotlib, Seaborn

## Author

Salamat Tolibaev

Data Scientist / ML Engineer
