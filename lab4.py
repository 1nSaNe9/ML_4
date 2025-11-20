# Import the dataset into your code
from ucimlrepo import fetch_ucirepo

# fetch dataset
adult = fetch_ucirepo(id=2)

# data (as pandas dataframes)
X = adult.data.features
y = adult.data.targets

# metadata
print(adult.metadata)

# variable information
print(adult.variables)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Обработка данных
X_processed = X.copy()

# Заполнение пропущенных значений (если есть)
X_processed = X_processed.fillna('Unknown')

# Кодируем категориальные признаки
categorical_columns = X_processed.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    X_processed[col] = le.fit_transform(X_processed[col].astype(str))
    label_encoders[col] = le

# Кодируем целевую переменную
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y.values.ravel())

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_encoded, test_size=0.2, random_state=42
)

# Преобразование в массив
y_train = np.array(y_train).ravel()
y_test = np.array(y_test).ravel()

# Масштабирование признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучение моделей с увеличенным количеством итераций
# Перцептрон
perceptron = Perceptron(max_iter=2000, random_state=42)
perceptron.fit(X_train_scaled, y_train)
y_pred_perceptron = perceptron.predict(X_test_scaled)
accuracy_perceptron = accuracy_score(y_test, y_pred_perceptron)

# MLPClassifier с базовыми параметрами и увеличенным max_iter
mlp = MLPClassifier(max_iter=2000, random_state=42)
mlp.fit(X_train_scaled, y_train)
y_pred_mlp = mlp.predict(X_test_scaled)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)

print(f"Точность Perceptron: {accuracy_perceptron:.4f}")
print(f"Точность MLPClassifier: {accuracy_mlp:.4f}")

# УПРОЩЕННЫЙ GridSearch для ускорения работы
param_grid = {
    'hidden_layer_sizes': [(50,), (100,)],  # Уменьшил количество вариантов
    'activation': ['relu', 'tanh'],
    'solver': ['adam'],  # Оставил только adam для скорости
    'alpha': [0.0001, 0.001],
    'learning_rate_init': [0.001, 0.01]
}

print("Запуск GridSearch... Это может занять несколько минут...")

grid_search = GridSearchCV(MLPClassifier(max_iter=1000, random_state=42),
                           param_grid,
                           scoring='accuracy',
                           cv=3,  # Уменьшил количество фолдов
                           n_jobs=-1,
                           verbose=1)  # Добавил прогресс бар

grid_search.fit(X_train_scaled, y_train)

print("Лучшие параметры:", grid_search.best_params_)
print("Лучшая точность на валидационной выборке:", grid_search.best_score_)

# Оценка на тестовой выборке с лучшими параметрами
best_mlp = grid_search.best_estimator_
y_pred_best = best_mlp.predict(X_test_scaled)
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"Точность лучшей модели на тестовой выборке: {accuracy_best:.4f}")

# БЫСТРАЯ визуализация - только основные результаты
results = pd.DataFrame(grid_search.cv_results_)

plt.figure(figsize=(10, 6))
# Показываем только среднюю точность по экспериментам
plt.plot(results['mean_test_score'].values, marker='o', linewidth=2)
plt.xlabel('Номер комбинации параметров')
plt.ylabel('Средняя точность')
plt.title('Результаты GridSearch (упрощенный набор параметров)')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Сохраняем график вместо показа (быстрее)
plt.savefig('gridsearch_results.png', dpi=150, bbox_inches='tight')
print("График сохранен как 'gridsearch_results.png'")

# Показываем только топ-5 результатов для экономии времени
top_results = results.nlargest(5, 'mean_test_score')[['mean_test_score', 'params']]
print("\nТоп-5 результатов:")
for i, (idx, row) in enumerate(top_results.iterrows(), 1):
    print(f"{i}. Точность: {row['mean_test_score']:.4f}")
    print(f"   Параметры: {row['params']}\n")

# Альтернатива: быстрый график без сохранения
plt.show()