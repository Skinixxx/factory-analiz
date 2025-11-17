#!/usr/bin/env python3
"""
rnn_experiments_report.py
Простой и надёжный скрипт для сравнительного теста RNN/LSTM/GRU + генерация PDF-отчёта.
По умолчанию сохраняет все временные изображения в папке temp_images.
Опционально можно удалить их флагом --clean.
"""

import os
import shutil
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from io import BytesIO
import logging
import sys

# ML & audio
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer

# PDF
from fpdf import FPDF
import matplotlib.font_manager as fm

# Настройка логов
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# -----------------------
# Полезные утилиты
# -----------------------
def ensure_temp_dir(path='temp_images'):
    os.makedirs(path, exist_ok=True)
    return path

def cleanup_temp_dir(path='temp_images'):
    try:
        if os.path.exists(path):
            shutil.rmtree(path)
            logging.info(f"Временная папка {path} удалена.")
    except Exception as e:
        logging.warning(f"Не удалось удалить временную папку {path}: {e}")

def save_figure(fig, path, close=True):
    fig.savefig(path, dpi=150, bbox_inches='tight')
    if close:
        plt.close(fig)

# -----------------------
# PDF-класс с поддержкой Unicode
# -----------------------
class PDFReport(FPDF):
    def __init__(self, font_path=None):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        # Подключаем шрифт Unicode (DejaVuSans обычно доступен)
        if font_path is None:
            try:
                font_path = fm.findfont("DejaVu Sans")
            except Exception:
                font_path = None
        if font_path and os.path.exists(font_path):
            try:
                self.add_font("DejaVu", "", font_path, uni=True)
                self.default_font = "DejaVu"
            except Exception as e:
                logging.warning(f"Не удалось подключить TTF-шрифт: {e}")
                self.default_font = "Arial"
        else:
            self.default_font = "Arial"
        self.set_font(self.default_font, size=12)

    def header(self):
        self.set_font(self.default_font, 'B', 16)
        self.cell(0, 10, "Отчёт по экспериментам: RNN-архитектуры", 0, 1, 'C')
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font(self.default_font, '', 9)
        self.cell(0, 10, f'Страница {self.page_no()}', 0, 0, 'C')

    def add_title_page(self, title, subtitle, meta_text):
        self.add_page()
        self.set_font(self.default_font, 'B', 20)
        self.ln(20)
        self.cell(0, 10, title, 0, 1, 'C')
        self.ln(6)
        self.set_font(self.default_font, '', 12)
        self.multi_cell(0, 8, subtitle, align='C')
        self.ln(10)
        self.set_font(self.default_font, '', 10)
        self.multi_cell(0, 6, meta_text, align='C')

    def add_section(self, heading, body=None):
        self.add_page()
        self.set_font(self.default_font, 'B', 14)
        self.cell(0, 10, heading, 0, 1, 'L')
        self.ln(2)
        if body:
            self.set_font(self.default_font, '', 11)
            self.multi_cell(0, 6, body)
            self.ln(3)

    def add_table(self, headers, rows):
        self.set_font(self.default_font, 'B', 11)
        effective_width = self.w - 2*self.l_margin
        col_w = effective_width / len(headers)
        # header
        for h in headers:
            self.cell(col_w, 8, str(h), 1, 0, 'C')
        self.ln()
        self.set_font(self.default_font, '', 10)
        for row in rows:
            for item in row:
                self.cell(col_w, 7, str(item), 1, 0, 'C')
            self.ln()

    def add_image(self, img_path, w=180):
        if os.path.exists(img_path):
            try:
                self.image(img_path, x=10, w=w)
                self.ln(4)
            except Exception as e:
                self.set_font(self.default_font, '', 10)
                self.multi_cell(0, 6, f"Не удалось вставить изображение {img_path}: {e}")

# -----------------------
# Данные (демо или замените на свою загрузку)
# -----------------------
def load_demo_data(num_samples=800, seq_len=100, n_features=40, n_classes=5, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(num_samples, seq_len, n_features)).astype(np.float32)
    y = rng.randint(0, n_classes, size=(num_samples,))
    logging.info("Созданы демонстрационные данные.")
    return X, y, n_classes

# -----------------------
# Модель
# -----------------------
def create_model(model_type='lstm', input_shape=(100,40), num_classes=5,
                 units=128, num_layers=1, dropout=0.2, bidirectional=False, lr=1e-3):
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))
    for i in range(num_layers):
        return_seq = (i < num_layers - 1)
        if model_type == 'lstm':
            core = layers.LSTM(units, return_sequences=return_seq, dropout=dropout)
        elif model_type == 'gru':
            core = layers.GRU(units, return_sequences=return_seq, dropout=dropout)
        elif model_type == 'rnn':
            core = layers.SimpleRNN(units, return_sequences=return_seq, dropout=dropout)
        else:
            raise ValueError("Unsupported model_type")
        if bidirectional:
            core = layers.Bidirectional(core)
        model.add(core)
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# -----------------------
# Эксперименты и отчётность
# -----------------------
def run_experiments(X, y, num_classes, experiments, out_dir='temp_images', epochs=8):
    ensure_temp_dir(out_dir)
    results = []
    histories = []
    cm_files = []

    # train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    for idx, cfg in enumerate(experiments):
        logging.info(f"Эксперимент {idx+1}/{len(experiments)}: {cfg['name']}")
        model = create_model(model_type=cfg['type'],
                             input_shape=X_train.shape[1:],
                             num_classes=num_classes,
                             units=cfg.get('units', 128),
                             num_layers=cfg.get('layers', 1),
                             dropout=cfg.get('dropout', 0.2),
                             bidirectional=cfg.get('bidirectional', False),
                             lr=cfg.get('lr', 1e-3))
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=epochs, batch_size=32, verbose=0)
        histories.append(history.history)

        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        macro = report.get('macro avg', {'precision':0,'recall':0,'f1-score':0})

        # ROC-AUC per class (fallback 0 if cannot compute)
        lb = LabelBinarizer()
        y_test_bin = lb.fit_transform(y_test)
        if y_test_bin.ndim == 1:
            y_test_bin = np.vstack([1 - y_test_bin, y_test_bin]).T
        roc_auc = []
        for c in range(num_classes):
            try:
                fpr, tpr, _ = roc_curve(y_test_bin[:, c], y_pred_proba[:, c])
                roc_auc.append(auc(fpr, tpr))
            except Exception:
                roc_auc.append(0.0)
        macro_auc = float(np.mean(roc_auc))

        results.append({
            'config': cfg,
            'accuracy': float(acc),
            'precision': float(macro['precision']),
            'recall': float(macro['recall']),
            'f1': float(macro['f1-score']),
            'auc': macro_auc
        })

        # confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Предсказанные метки')
        ax.set_ylabel('Истинные метки')
        ax.set_title(f'Матрица ошибок: {cfg["name"]}')
        cm_path = os.path.join(out_dir, f'cm_{idx+1}.png')
        save_figure(fig, cm_path)
        cm_files.append(cm_path)

    # make training plots summary (loss + acc)
    fig, axes = plt.subplots(1,2, figsize=(12,4))
    for i, h in enumerate(histories):
        if 'loss' in h:
            axes[0].plot(h['loss'], label=experiments[i]['name'])
        if 'accuracy' in h:
            axes[1].plot(h['accuracy'], label=experiments[i]['name'])
    axes[0].set_title('Loss (train)')
    axes[1].set_title('Accuracy (train)')
    axes[0].legend(); axes[1].legend()
    train_plot_path = os.path.join(out_dir, 'training_summary.png')
    save_figure(fig, train_plot_path)

    # metric comparison bar chart
    fig, ax = plt.subplots(figsize=(10,5))
    names = [r['config']['name'] for r in results]
    accs = [r['accuracy'] for r in results]
    f1s = [r['f1'] for r in results]
    x = np.arange(len(results))
    width = 0.35
    ax.bar(x - width/2, accs, width, label='Accuracy')
    ax.bar(x + width/2, f1s, width, label='F1-score')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_title('Сравнение метрик')
    ax.legend()
    metrics_path = os.path.join(out_dir, 'metrics_comparison.png')
    save_figure(fig, metrics_path)

    return results, cm_files, train_plot_path, metrics_path

# -----------------------
# Отчёт в PDF
# -----------------------
def build_pdf_report(results, cm_files, train_plot, metrics_plot, out_pdf='RNN_Report.pdf'):
    title = "Сравнение RNN/LSTM/GRU архитектур"
    subtitle = "Автоматически сгенерированный отчёт по экспериментам.\nВ отчёте показаны метрики, матрицы ошибок и графики обучения."
    meta = f"Дата: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}\nЭкспериментов: {len(results)}"

    pdf = PDFReport()
    pdf.add_title_page(title=title, subtitle=subtitle, meta_text=meta)

    # Summary table
    pdf.add_section("Сводные результаты по экспериментам")
    headers = ["#", "Архитектура", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    rows = []
    for i, r in enumerate(results, start=1):
        rows.append([i, r['config']['name'],
                     f"{r['accuracy']:.3f}",
                     f"{r['precision']:.3f}",
                     f"{r['recall']:.3f}",
                     f"{r['f1']:.3f}",
                     f"{r['auc']:.3f}"])
    pdf.add_table(headers, rows)

    # Best model highlight
    best_idx = int(np.argmax([r['f1'] for r in results]))
    best = results[best_idx]
    pdf.add_section("Лучшая модель", f"Лучшая модель по F1: {best['config']['name']}\n"
                                    f"F1 = {best['f1']:.3f}, Accuracy = {best['accuracy']:.3f}")

    # add training + metrics images
    pdf.add_section("Графики обучения")
    pdf.add_image(train_plot, w=180)
    pdf.add_section("Сравнение метрик")
    pdf.add_image(metrics_plot, w=180)

    # confusion matrices
    pdf.add_section("Матрицы ошибок")
    for i, cmf in enumerate(cm_files):
        pdf.add_section(f"Матрица ошибок: {results[i]['config']['name']}")
        pdf.add_image(cmf, w=160)

    # Save PDF
    pdf.output(out_pdf)
    logging.info(f"PDF-отчёт сохранён: {out_pdf}")
    return out_pdf

# -----------------------
# CLI / main
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="RNN experiments and PDF report generator")
    parser.add_argument("--epochs", type=int, default=8, help="Число эпох обучения (по умолчанию 8)")
    parser.add_argument("--out_pdf", type=str, default=None, help="Имя выходного PDF (по умолчанию генерируется автоматически)")
    parser.add_argument("--clean", action="store_true", help="Удалить temp_images после выполнения (по умолчанию НЕ удалять)")
    args = parser.parse_args()

    logging.info("Запуск экспериментов...")

    # Демонстрационные данные: замените на вашу загрузку (MFCC) при необходимости
    X, y, n_classes = load_demo_data(num_samples=600, seq_len=100, n_features=40, n_classes=5)

    # Список экспериментов (упрощённый)
    experiments = [
        {'name': 'Простая RNN', 'type': 'rnn', 'units': 64, 'layers': 1},
        {'name': 'LSTM', 'type': 'lstm', 'units': 128, 'layers': 1},
        {'name': 'GRU', 'type': 'gru', 'units': 128, 'layers': 1},
        {'name': 'Двунаправленная LSTM', 'type': 'lstm', 'units': 64, 'layers': 2, 'bidirectional': True},
        {'name': 'Двунаправленная GRU + Dropout', 'type': 'gru', 'units': 64, 'layers': 2, 'bidirectional': True, 'dropout': 0.3},
    ]

    temp_dir = ensure_temp_dir('temp_images')

    try:
        results, cm_files, train_plot, metrics_plot = run_experiments(X, y, n_classes, experiments, out_dir=temp_dir, epochs=args.epochs)
        if args.out_pdf:
            pdf_name = args.out_pdf
        else:
            pdf_name = f'RNN_Experiment_Report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        build_pdf_report(results, cm_files, train_plot, metrics_plot, out_pdf=pdf_name)
        logging.info("Готово. Все изображения сохранены в папке: %s", temp_dir)
    finally:
        if args.clean:
            cleanup_temp_dir(temp_dir)
        else:
            logging.info("temp_images сохранена (не удаляется). Если нужно удалить — запустите с флагом --clean")

if __name__ == "__main__":
    main()
