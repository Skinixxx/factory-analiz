import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
import librosa
from tqdm import tqdm
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Для избежания конфликтов с GUI

# Новые импорты для PDF
from fpdf import FPDF
import base64
from io import BytesIO

class PDFReport(FPDF):
    """Класс для создания PDF-отчета на русском языке"""
    
    def __init__(self):
        super().__init__()
        # Добавляем поддержку кириллицы
        self.add_font('DejaVu', '', 'DejaVuSansCondensed.ttf', uni=True)
        self.add_font('DejaVu', 'B', 'DejaVuSansCondensed-Bold.ttf', uni=True)
        self.set_auto_page_break(auto=True, margin=15)
        
    def header(self):
        """Верхний колонтитул"""
        self.set_font('DejaVu', 'B', 16)
        self.cell(0, 10, 'Отчет по экспериментам RNN-архитектур', 0, 1, 'C')
        self.ln(5)
        
    def footer(self):
        """Нижний колонтитул"""
        self.set_y(-15)
        self.set_font('DejaVu', 'I', 8)
        self.cell(0, 10, f'Страница {self.page_no()}', 0, 0, 'C')
        
    def chapter_title(self, title):
        """Заголовок раздела"""
        self.set_font('DejaVu', 'B', 14)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(2)
        
    def chapter_body(self, body):
        """Текст раздела"""
        self.set_font('DejaVu', '', 12)
        self.multi_cell(0, 8, body)
        self.ln()
        
    def add_table(self, data, headers):
        """Добавление таблицы"""
        self.set_font('DejaVu', 'B', 10)
        
        # Расчет ширины колонок
        col_width = self.w / (len(headers) + 1)
        
        # Заголовки
        for header in headers:
            self.cell(col_width, 10, header, border=1, align='C')
        self.ln()
        
        # Данные
        self.set_font('DejaVu', '', 10)
        for row in data:
            for item in row:
                self.cell(col_width, 8, str(item), border=1, align='C')
            self.ln()
            
    def add_image(self, image_path, width=180):
        """Добавление изображения"""
        if os.path.exists(image_path):
            self.image(image_path, x=10, y=None, w=width)
            self.ln(5)

def matplotlib_to_img():
    """Конвертирует текущий график matplotlib в base64"""
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    return base64.b64encode(image_png).decode('utf-8')

def save_plot_to_file(plt, filename):
    """Сохраняет график в файл"""
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

# Функции для обработки данных (остаются без изменений)
def extract_features(file_path, n_mfcc=40, fixed_length=100):
    try:
        audio, sr = librosa.load(file_path, sr=22050)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        
        if mfcc.shape[1] > fixed_length:
            mfcc = mfcc[:, :fixed_length]
        else:
            pad_width = fixed_length - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
        return mfcc.T
    except Exception as e:
        print(f"Ошибка при обработке файла {file_path}: {e}")
        return None

def load_demo_data():
    """Создание демонстрационных данных"""
    num_samples = 800
    sequence_length = 100
    num_features = 40
    num_classes = 5
    
    X = np.random.random((num_samples, sequence_length, num_features))
    y = np.random.randint(0, num_classes, num_samples)
    
    print("Используются демонстрационные данные.")
    return X, y, num_classes

def create_model(model_type='lstm', units=128, num_layers=1, dropout_rate=0.2,
                 recurrent_dropout=0.2, bidirectional=False, learning_rate=0.001,
                 input_shape=(100, 40), num_classes=5):
    
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))
    
    for i in range(num_layers):
        return_sequences = (i < num_layers - 1)
        
        if model_type == 'lstm':
            layer = layers.LSTM(units, return_sequences=return_sequences, 
                               dropout=dropout_rate, recurrent_dropout=recurrent_dropout)
        elif model_type == 'gru':
            layer = layers.GRU(units, return_sequences=return_sequences,
                              dropout=dropout_rate, recurrent_dropout=recurrent_dropout)
        elif model_type == 'rnn':
            layer = layers.SimpleRNN(units, return_sequences=return_sequences,
                                    dropout=dropout_rate, recurrent_dropout=recurrent_dropout)
        else:
            raise ValueError("Неизвестный тип модели")
        
        if bidirectional:
            layer = layers.Bidirectional(layer)
        model.add(layer)
    
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def main():
    print("=== ЭКСПЕРИМЕНТАЛЬНОЕ СРАВНЕНИЕ RNN-АРХИТЕКТУР ===")
    
    # Создаем папку для временных изображений
    os.makedirs('temp_images', exist_ok=True)
    image_files = []
    
    # Загрузка данных
    X, y, num_classes = load_demo_data()
    
    # Разделение данных
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Параметры экспериментов
    experiments = [
        {'model_type': 'rnn', 'units': 64, 'num_layers': 1, 'bidirectional': False, 'name': 'Простая RNN'},
        {'model_type': 'lstm', 'units': 128, 'num_layers': 1, 'bidirectional': False, 'name': 'LSTM'},
        {'model_type': 'gru', 'units': 128, 'num_layers': 1, 'bidirectional': False, 'name': 'GRU'},
        {'model_type': 'lstm', 'units': 64, 'num_layers': 2, 'bidirectional': True, 'name': 'Двунаправленная LSTM'},
        {'model_type': 'gru', 'units': 64, 'num_layers': 2, 'bidirectional': True, 'dropout_rate': 0.3, 'name': 'Двунаправленная GRU с Dropout'},
    ]
    
    # Обучение и оценка моделей
    results = []
    history_dict = {}
    
    print("\n--- Начало экспериментов ---")
    for i, config in enumerate(experiments):
        print(f"\nЭксперимент {i+1}: {config['name']}")
        
        model = create_model(
            model_type=config['model_type'],
            units=config['units'],
            num_layers=config['num_layers'],
            bidirectional=config.get('bidirectional', False),
            dropout_rate=config.get('dropout_rate', 0.2),
            input_shape=X_train.shape[1:],
            num_classes=num_classes
        )
        
        # Обучение
        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=15,
            validation_data=(X_val, y_val),
            verbose=0
        )
        history_dict[i] = history.history
        
        # Оценка
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Метрики
        report = classification_report(y_test, y_pred, output_dict=True)
        macro_avg = report['macro avg']
        
        # ROC-AUC
        lb = LabelBinarizer()
        y_test_bin = lb.fit_transform(y_test)
        roc_auc = {}
        for j in range(num_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, j], y_pred_proba[:, j])
            roc_auc[j] = auc(fpr, tpr)
        macro_auc = np.mean(list(roc_auc.values()))
        
        results.append({
            'config': config,
            'test_accuracy': test_accuracy,
            'test_precision': macro_avg['precision'],
            'test_recall': macro_avg['recall'],
            'test_f1': macro_avg['f1-score'],
            'test_auc': macro_auc
        })
        
        # Матрица ошибок для отчета
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Матрица ошибок: {config["name"]}')
        plt.ylabel('Истинные метки')
        plt.xlabel('Предсказанные метки')
        cm_filename = f'temp_images/cm_exp_{i+1}.png'
        save_plot_to_file(plt, cm_filename)
        image_files.append(cm_filename)
    
    # Создание графиков обучения
    plt.figure(figsize=(15, 10))
    
    # Графики потерь
    plt.subplot(2, 2, 1)
    for i, config in enumerate(experiments):
        plt.plot(history_dict[i]['loss'], label=f'{config["name"]}')
    plt.title('Функция потерь на обучении')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Графики точности
    plt.subplot(2, 2, 2)
    for i, config in enumerate(experiments):
        plt.plot(history_dict[i]['accuracy'], label=f'{config["name"]}')
    plt.title('Точность на обучении')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Графики потерь на валидации
    plt.subplot(2, 2, 3)
    for i, config in enumerate(experiments):
        plt.plot(history_dict[i]['val_loss'], label=f'{config["name"]}')
    plt.title('Функция потерь на валидации')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Графики точности на валидации
    plt.subplot(2, 2, 4)
    for i, config in enumerate(experiments):
        plt.plot(history_dict[i]['val_accuracy'], label=f'{config["name"]}')
    plt.title('Точность на валидации')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    training_plot_filename = 'temp_images/training_plots.png'
    save_plot_to_file(plt, training_plot_filename)
    image_files.append(training_plot_filename)
    
    # Сравнительная визуализация метрик
    metrics_names = ['Точность', 'Precision', 'Recall', 'F1-score', 'ROC-AUC']
    metrics_values = [
        [res['test_accuracy'] for res in results],
        [res['test_precision'] for res in results],
        [res['test_recall'] for res in results],
        [res['test_f1'] for res in results],
        [res['test_auc'] for res in results]
    ]
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(experiments))
    width = 0.15
    
    for i, (name, values) in enumerate(zip(metrics_names, metrics_values)):
        plt.bar(x + i*width, values, width, label=name)
    
    plt.xlabel('Эксперименты')
    plt.ylabel('Значения метрик')
    plt.title('Сравнение метрик по экспериментам')
    plt.xticks(x + width*2, [f'Эксп.{i+1}' for i in range(len(experiments))])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    metrics_plot_filename = 'temp_images/metrics_comparison.png'
    save_plot_to_file(plt, metrics_plot_filename)
    image_files.append(metrics_plot_filename)
    
    # СОЗДАНИЕ PDF-ОТЧЕТА
    print("\n--- Создание PDF-отчета ---")
    pdf = PDFReport()
    pdf.add_page()
    
    # Титульная страница
    pdf.set_font('DejaVu', 'B', 20)
    pdf.cell(0, 50, 'ОТЧЕТ ПО ЭКСПЕРИМЕНТАМ', 0, 1, 'C')
    pdf.set_font('DejaVu', '', 14)
    pdf.cell(0, 10, 'Сравнение RNN-архитектур для акустической классификации', 0, 1, 'C')
    pdf.ln(20)
    pdf.set_font('DejaVu', '', 12)
    pdf.cell(0, 10, f'Дата генерации: {datetime.now().strftime("%d.%m.%Y %H:%M")}', 0, 1, 'C')
    pdf.cell(0, 10, f'Количество экспериментов: {len(experiments)}', 0, 1, 'C')
    pdf.cell(0, 10, f'Размер dataset: {len(X)} samples', 0, 1, 'C')
    
    # Страница с результатами
    pdf.add_page()
    pdf.chapter_title('Результаты экспериментов')
    
    # Сводная таблица
    table_data = []
    for i, res in enumerate(results):
        table_data.append([
            f"Эксп. {i+1}",
            res['config']['name'],
            f"{res['test_accuracy']:.4f}",
            f"{res['test_precision']:.4f}",
            f"{res['test_recall']:.4f}",
            f"{res['test_f1']:.4f}",
            f"{res['test_auc']:.4f}"
        ])
    
    pdf.chapter_body('Сводная таблица метрик для всех архитектур:')
    pdf.add_table(table_data, ['№', 'Архитектура', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC'])
    
    # Находим лучшую модель
    best_idx = np.argmax([res['test_f1'] for res in results])
    best_model = results[best_idx]
    
    pdf.ln(10)
    pdf.chapter_title('Лучшая модель')
    pdf.chapter_body(
        f"Лучшие результаты показала архитектура: {best_model['config']['name']}\n"
        f"F1-score: {best_model['test_f1']:.4f}\n"
        f"Accuracy: {best_model['test_accuracy']:.4f}\n"
        f"ROC-AUC: {best_model['test_auc']:.4f}"
    )
    
    # Графики обучения
    pdf.add_page()
    pdf.chapter_title('Графики процесса обучения')
    pdf.chapter_body('Динамика изменения функции потерь и точности в процессе обучения:')
    pdf.add_image(training_plot_filename)
    
    # Сравнение метрик
    pdf.add_page()
    pdf.chapter_title('Сравнение метрик')
    pdf.chapter_body('Сравнительный анализ метрик для всех архитектур:')
    pdf.add_image(metrics_plot_filename)
    
    # Матрицы ошибок
    pdf.add_page()
    pdf.chapter_title('Матрицы ошибок')
    pdf.chapter_body('Матрицы ошибок для каждой тестируемой архитектуры:')
    
    for i, cm_file in enumerate(image_files[:-2]):  # Все кроме последних двух графиков
        if 'cm_exp' in cm_file:
            pdf.add_page()
            pdf.chapter_title(f'Матрица ошибок: {experiments[i]["name"]}')
            pdf.add_image(cm_file, width=160)
    
    # Сохранение PDF
    pdf_filename = f'RNN_Experiment_Report_{datetime.now().strftime("%Y%m%d_%H%M")}.pdf'
    pdf.output(pdf_filename)
    
    # Очистка временных файлов
    for file in image_files:
        if os.path.exists(file):
            os.remove(file)
    if os.path.exists('temp_images'):
        os.rmdir('temp_images')
    
    print(f"\n✅ Отчет успешно сохранен в файл: {pdf_filename}")
    print(f"📊 Протестировано архитектур: {len(experiments)}")
    print(f"🏆 Лучшая модель: {best_model['config']['name']} (F1-score: {best_model['test_f1']:.4f})")

if __name__ == "__main__":
    main()