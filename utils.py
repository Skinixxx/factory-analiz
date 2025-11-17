import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Создаем DataFrame с данными
data = {
    'Район': ['Абдулинский', 'Адамовский', 'Амбулакский', 'Александровский', 'Асажеевский',
             'Беляевский', 'Бугурусланский', 'Бузулукский', 'Гайский', 'Грачевский',
             'Домбаровский', 'Илекский', 'Кварянский', 'Красногвардейский', 'Кузенцакский',
             'Кузманаевский', 'Маневенский', 'Новорский', 'Новосергиевский', 'Октябрьский',
             'Оренбургский', 'Переконайский', 'Переволодский', 'Пономаревский', 'Самарский',
             'Саракашский', 'Святильский', 'Северный', 'Соль-Илецкий', 'Соломинский',
             'Ташлинский', 'Тощий', 'Тюльганский', 'Шарпинский', 'Ясненский'],
    'X4': [3163.3, 4337.1, 1982.7, 1171.1, 3387.7, 2564.1, 2546.7, 2398.4, 1368.6, 2980.2,
           2184.2, 2798.4, 1840.5, 3005.2, 5007.1, 2677.7, 2650.8, 5842.8, 5868.8, 4818.7,
           14470.7, 3573.6, 1364.6, 3857.1, 4484.6, 4405.1, 902.6, 2727.6, 341.3, 1106.8,
           5105.7, 1847.5, 2001.8, 3674.3, 122.1],
    'X6': [165, 328.3, 119.3, 83.5, 253.1, 189.5, 163.5, 159.5, 102.3, 222.7,
           131.3, 256.7, 139.3, 222.9, 353.3, 181.1, 204.2, 443.5, 344.2, 417.7,
           937.2, 224.1, 105.5, 288.2, 335.1, 329.9, 32.9, 203.3, 43.8, 90.3,
           343.1, 177.9, 149.4, 238.4, 9.1],
    'X7': [1919, 9537, 3543, 1611, 5822, 3679, 3647, 5666, 1135, 3340,
           2249, 5800, 2767, 5194, 8090, 2731, 2781, 13282, 12700, 5296,
           59234, 5431, 2974, 4842, 10120, 12330, 1027, 3368, 478, 1290,
           9010, 5539, 3510, 4934, 52],
    'X9': [235.6, 200.1, 169.4, 154.8, 209.4, 180.5, 250.1, 199, 245.2, 214.3,
           115.2, 167.6, 184.4, 231.3, 244.8, 189.2, 208.8, 153.3, 232.5, 166.8,
           234.1, 192.9, 198, 161.6, 173.9, 208.4, 113.5, 162.7, 191.3, 238.8,
           150.6, 148.9, 167, 258, 98.4],
    'X12': [4408, 6119, 5046, 5576, 5063, 5768, 5182, 6523, 4984, 6877,
            6182, 5238, 6802, 7125, 4379, 7808, 5500, 9153, 6792, 6434,
            13975, 8976, 5994, 6584, 6788, 5552, 7448, 6540, 4602, 5326,
            4672, 6331, 5953, 5880, 5047],
    'Класс': [None, 2, 1, 1, None, 1, None, 1, 1, None,
             1, None, 1, 2, 2, None, 1, 2, 2, None,
             None, None, 1, 2, 2, 2, 1, None, 1, 1,
             2, 1, 1, 2, 1]
}

df = pd.DataFrame(data)

# Определяем обучающие выборки для варианта 0
training_class1 = [3, 4, 6, 8, 9, 11, 13, 17, 23, 27, 29, 30, 32, 33, 35]  # индексы с 0
training_class2 = [2, 14, 15, 18, 19, 24, 25, 26, 31, 34]  # индексы с 0

# Подготовка данных для обучения
features = ['X4', 'X6', 'X7', 'X9', 'X12']

# Создаем массивы для обучающих данных
X_train = []
y_train = []

# Добавляем данные для класса 1
for idx in training_class1:
    # Индексы в DataFrame начинаются с 0, но в списке training_class1 индексы с 1
    X_train.append(df.loc[idx-1, features].values)
    y_train.append(1)

# Добавляем данные для класса 2
for idx in training_class2:
    X_train.append(df.loc[idx-1, features].values)
    y_train.append(2)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Данные для классификации (районы без указания класса)
unknown_indices = [i for i, val in enumerate(df['Класс']) if pd.isna(val)]
X_unknown = df.loc[unknown_indices, features].values
unknown_districts = df.loc[unknown_indices, 'Район'].values

print("=== ДИСКРИМИНАНТНЫЙ АНАЛИЗ РАЙОНОВ ОРЕНБУРГСКОЙ ОБЛАСТИ ===")
print(f"Количество обучающих образцов: {len(X_train)}")
print(f"Количество районов для классификации: {len(X_unknown)}")
print()

# Проверяем условие: число объектов в каждой обучающей выборке должно быть хотя бы на 2 больше чем число признаков
n_class1 = len(training_class1)
n_class2 = len(training_class2)
n_features = len(features)

print("ПРОВЕРКА УСЛОВИЙ:")
print(f"Количество признаков: {n_features}")
print(f"Количество объектов в классе 1: {n_class1}")
print(f"Количество объектов в классе 2: {n_class2}")
print(f"Условие выполняется для класса 1: {n_class1 >= n_features + 2}")
print(f"Условие выполняется для класса 2: {n_class2 >= n_features + 2}")
print()

# Нормализуем данные
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_unknown_scaled = scaler.transform(X_unknown)

# Создаем и обучаем модель LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_scaled, y_train)

# Прогнозируем классы для неизвестных районов
predictions = lda.predict(X_unknown_scaled)
probabilities = lda.predict_proba(X_unknown_scaled)

# Выводим результаты классификации
print("РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ:")
print("=" * 80)
for i, district in enumerate(unknown_districts):
    pred_class = predictions[i]
    prob_class1 = probabilities[i][0]
    prob_class2 = probabilities[i][1]
    
    print(f"{district:<20} -> Класс {pred_class} (вероятность: {max(prob_class1, prob_class2):.3f})")

print()
print("=" * 80)

# Анализ важности признаков
feature_importance = np.abs(lda.coef_[0])
feature_importance_df = pd.DataFrame({
    'Признак': features,
    'Важность': feature_importance
}).sort_values('Важность', ascending=False)

print("\nВАЖНОСТЬ ПРИЗНАКОВ:")
for _, row in feature_importance_df.iterrows():
    print(f"{row['Признак']}: {row['Важность']:.4f}")

# Визуализация результатов
plt.figure(figsize=(15, 10))

# 1. График распределения районов в пространстве дискриминантных функций
X_lda = lda.transform(X_train_scaled)
X_unknown_lda = lda.transform(X_unknown_scaled)

plt.subplot(2, 2, 1)
colors = ['red', 'blue']
for i, class_label in enumerate([1, 2]):
    mask = y_train == class_label
    plt.scatter(X_lda[mask, 0], np.zeros_like(X_lda[mask, 0]), 
                c=colors[i], label=f'Класс {class_label}', alpha=0.7)

for i, (x, pred) in enumerate(zip(X_unknown_lda[:, 0], predictions)):
    color = colors[pred-1]
    plt.scatter(x, 0, c=color, marker='*', s=150, 
               label=f'Класс {pred}' if i == 0 else "")

plt.xlabel('Дискриминантная функция 1')
plt.title('Распределение районов по классам')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Вероятности классификации
plt.subplot(2, 2, 2)
x_pos = np.arange(len(unknown_districts))
plt.bar(x_pos, np.max(probabilities, axis=1), 
        color=[colors[p-1] for p in predictions], alpha=0.7)
plt.xticks(x_pos, [d[:10] + '...' for d in unknown_districts], rotation=45)
plt.ylabel('Вероятность правильной классификации')
plt.title('Вероятности отнесения к классам')
plt.ylim(0.5, 1.0)

# 3. Важность признаков
plt.subplot(2, 2, 3)
plt.barh(feature_importance_df['Признак'], feature_importance_df['Важность'])
plt.xlabel('Важность признака')
plt.title('Важность признаков в дискриминантном анализе')

# 4. Статистика по классам
plt.subplot(2, 2, 4)
class_counts = pd.Series(predictions).value_counts().sort_index()
plt.pie(class_counts.values, labels=[f'Класс {i}' for i in class_counts.index], 
        autopct='%1.1f%%', colors=['red', 'blue'])
plt.title('Распределение классифицированных районов')

plt.tight_layout()
plt.show()

# Экономическая интерпретация результатов
print("\nЭКОНОМИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ:")
print("=" * 80)

# Анализ средних значений по классам
class1_data = df[df['Класс'] == 1][features]
class2_data = df[df['Класс'] == 2][features]

print("\nСРЕДНИЕ ЗНАЧЕНИЯ ПО КЛАССАМ:")
print("Класс 1:")
for feature in features:
    mean_val = class1_data[feature].mean()
    print(f"  {feature}: {mean_val:.1f}")

print("\nКласс 2:")
for feature in features:
    mean_val = class2_data[feature].mean()
    print(f"  {feature}: {mean_val:.1f}")

print("\nХАРАКТЕРИСТИКИ КЛАССОВ:")
print("Класс 1 - Районы с более низкими инвестициями и развитием:")
print("  • Ниже инвестиции в жилищное хозяйство")
print("  • Меньший ввод жилых домов")
print("  • Более высокий удельный вес убыточных организаций")

print("\nКласс 2 - Районы с более высокими инвестициями и развитием:")
print("  • Выше инвестиции в жилищное хозяйство") 
print("  • Больший ввод жилых домов")
print("  • Ниже удельный вес убыточных организаций")

# Дополнительный анализ
print("\nДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ:")
print(f"Точность на обучающей выборке: {lda.score(X_train_scaled, y_train):.3f}")
print(f"Общая ковариационная матрица оценена успешно")
print(f"Модель использует {lda.n_components_} дискриминантных компонент")

# Сохраняем результаты в файл
results_df = df.copy()
for i, idx in enumerate(unknown_indices):
    results_df.loc[idx, 'Класс'] = predictions[i]
    results_df.loc[idx, 'Вероятность'] = max(probabilities[i])

results_df.to_csv('результаты_классификации.csv', index=False, encoding='utf-8-sig')
print("\nРезультаты сохранены в файл 'результаты_классификации.csv'")