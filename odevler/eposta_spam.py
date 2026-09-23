# -*- coding: utf-8 -*-
"""
===============================================================================
MINI MACHINE LEARNING PROJESI - E-POSTA SPAM TAHMINI (TAMAMLANMIŞ ÖDEV)
===============================================================================
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from colorama import Fore, Style
import matplotlib.pyplot as plt

# Sklearn Yapay Zeka Kütüphaneleri
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Raporlama Kütüphanesi
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


class AppState:
    def __init__(self):
        self.csv_path = None
        self.raw_df = None
        self.df = None
        self.target_column = "spam"
        self.feature_columns = []
        self.best_model = None
        self.best_model_name = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.y_pred = None
        self.model_results = []
        self.preprocessing_completed = False


def print_header(title: str) -> None:
    print(Fore.YELLOW + "\n" + "=" * 78)
    print(f"--- {title} ---")
    print("=" * 78 + Style.RESET_ALL)


def normalize_column_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


# STEP 1: Veri Yükleme
def load_csv(state: AppState) -> None:
    # Dosya yolu hocanın kılavuzuna göre sabitlendi
    path = Path("odevler/ornek_email_spam.csv")
    if not path.exists():
        print(Fore.RED + f"\nHATA: {path.name} dosyası klasörde bulunamadı!" + Style.RESET_ALL)
        return

    state.csv_path = path
    state.raw_df = pd.read_csv(path)
    state.raw_df.columns = [normalize_column_name(col) for col in state.raw_df.columns]
    state.df = state.raw_df.copy(deep=True)
    state.feature_columns = [col for col in state.df.columns if col != state.target_column]

    print_header("CSV BAŞARIYLA YÜKLENDİ")
    print(f"Dosya Adı: {path.name} | Satır Sayısı: {len(state.df)} | Eksik Veri: {state.df.isna().sum().sum()}")
    state.preprocessing_completed = False


# STEP 2: Veri Temizleme (Hocanın Beklediği SimpleImputer Yapısı)
def clean_data(state: AppState) -> None:
    if state.df is None:
        print(Fore.RED + "\nÖnce veri yüklemelisiniz!" + Style.RESET_ALL)
        return

    print_header("VERİ TEMİZLEME VE ÖN İŞLEME")
    # Eksik verileri ortalamayla dolduruyoruz (Ders 3 konusu)
    imputer = SimpleImputer(strategy="mean")
    state.df[state.feature_columns] = imputer.fit_transform(state.df[state.feature_columns])

    # Yinelenen satırları düşürüyoruz
    state.df.drop_duplicates(inplace=True)
    state.preprocessing_completed = True
    print(Fore.GREEN + "-> Eksik veriler ortalamayla dolduruldu ve temizlendi!" + Style.RESET_ALL)


# STEP 3: Veri Önizleme
def preview_data(state: AppState) -> None:
    if state.df is None: return
    print_header("VERİ ÖNİZLEME (İLK 5 SATIR)")
    print(state.df.head())


# STEP 4: Veri Kalitesi Analizi
def analyze_quality(state: AppState) -> None:
    if state.df is None: return
    print_header("VERİ KALİTESİ ANALİZİ")
    print("Sütun bazlı eksik veri sayıları:")
    print(state.df.isna().sum())


# STEP 6: Model Eğitimi (Pipeline, Logistic Regression, Decision Tree, Random Forest)
def train_models(state: AppState) -> None:
    if not state.preprocessing_completed:
        print(Fore.RED + "\nÖnce 2 numaralı seçenekle veriyi temizlemelisiniz!" + Style.RESET_ALL)
        return

    print_header("MODELLERİN EĞİTİLMESİ (PIPELINE)")
    X = state.df[state.feature_columns]
    y = state.df[state.target_column].astype(int)

    # Train-Test Bölmesi (Hocanın standart %20 test ayarı)
    state.X_train, state.X_test, state.y_train, state.y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Modelleri tanımlıyoruz (Ders 4 konusu)
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    state.model_results = []
    best_f1 = -1

    for name, model in models.items():
        # Uçtan uca Pipeline kuruyoruz (StandardScaler + Model)
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])

        pipe.fit(state.X_train, state.y_train)
        preds = pipe.predict(state.X_test)

        acc = accuracy_score(state.y_test, preds)
        f1 = f1_score(state.y_test, preds, average='macro')

        state.model_results.append({"Model": name, "Accuracy": acc, "F1-Score": f1, "Pipeline": pipe})
        print(f"- {name} eğitildi. Model Başarısı (Accuracy): {round(acc, 2)}")

        # En iyi modeli F1 skoruna göre seçiyoruz
        if f1 > best_f1:
            best_f1 = f1
            state.best_model = pipe
            state.best_model_name = name
            state.y_pred = preds


# STEP 7: Sonuç Karşılaştırma
def compare_results(state: AppState) -> None:
    print_header("MODEL SONUÇLARI KARŞILAŞTIRMA")
    for res in state.model_results:
        print(f"Model: {res['Model']} | Accuracy: {round(res['Accuracy'], 2)} | F1-Score: {round(res['F1-Score'], 2)}")


# STEP 8: Raporlama (TXT ve PDF Raporu Oluşturma)
def generate_reports(state: AppState) -> None:
    if state.best_model is None: return
    print_header("RAPORLARIN KAYDEDİLMESİ")

    # 1. TXT Raporu
    with open("sonuclar.txt", "w", encoding="utf-8") as f:
        f.write(f"En Başarılı Model: {state.best_model_name}\n")
        f.write(classification_report(state.y_test, state.y_pred))
    print("-> 'sonuclar.txt' başarıyla oluşturuldu.")

    # 2. PDF Raporu (Hocanın reportlab isteği)
    doc = SimpleDocTemplate("Spam_Classification_Report.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    story = [
        Paragraph(f"<b>Yapay Zeka Bootcamp - E-Posta Spam Tahmin Raporu</b>", styles['Title']),
        Spacer(1, 12),
        Paragraph(f"En Basarili Model: {state.best_model_name}", styles['Heading2']),
        Spacer(1, 12),
        Paragraph("Detayli Metrikler sonuclar.txt dosyasına başarıyla kaydedilmiştir.", styles['Normal'])
    ]
    doc.build(story)
    print("-> 'Spam_Classification_Report.pdf' başarıyla oluşturuldu.")


# STEP 9: Confusion Matrix Grafiği (Matplotlib)
def save_confusion_matrix(state: AppState) -> None:
    if state.y_test is None: return
    print_header("CONFUSION MATRIX GRAFİĞİNİN KAYDEDİLMESİ")
    cm = confusion_matrix(state.y_test, state.y_pred)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {state.best_model_name}")
    plt.colorbar()
    plt.ylabel('Gerçek Sınıf')
    plt.xlabel('Tahmin Edilen Sınıf')
    plt.savefig("confusion_matrix.png")
    plt.close()
    print(Fore.GREEN + "-> 'confusion_matrix.png' başarıyla kaydedildi!" + Style.RESET_ALL)


# STEP 10: Canlı Tahmin (Inference Aşaması)
def predict_new(state: AppState) -> None:
    if state.best_model is None:
        print(Fore.RED + "\nÖnce modelleri eğitmelisiniz!" + Style.RESET_ALL)
        return
    print_header("YENİ E-POSTA İÇİN SPAM TAHMİNİ")
    try:
        ks = float(input("Kelime Sayısı: "))
        ls = float(input("Link Sayısı: "))
        bho = float(input("Büyük Harf Oranı (0-1 arası): "))
        sk = float(input("Şüpheli Kelime Sayısı: "))
        gp = float(input("Gönderici Puanı (0-100 arası): "))
        ev = float(input("Ek Var mı? (0 veya 1): "))

        yeni_veri = pd.DataFrame([[ks, ls, bho, sk, gp, ev]], columns=state.feature_columns)
        tahmin = state.best_model.predict(yeni_veri)[0]

        if tahmin == 1:
            print(Fore.RED + "\nSONUÇ: Bu e-posta %100 SPAM olarak tespit edildi!" + Style.RESET_ALL)
        else:
            print(Fore.GREEN + "\nSONUÇ: Bu e-posta GÜVENLİDIR." + Style.RESET_ALL)
    except ValueError:
        print("\nLütfen sadece sayısal değerler giriniz.")


# ANA MENÜ AKIŞI
def main():
    state = AppState()
    while True:
        print(Fore.CYAN + "\n" + "=" * 45)
        print("=== AKILLI ML PIPELINE MENÜSÜ ===")
        print("=" * 45 + Style.RESET_ALL)
        print("1 - CSV Dosyasını Yükle")
        print("2 - Veri Temizle & Kaydet (SimpleImputer)")
        print("3 - Veriyi Önizle")
        print("4 - Veri Kalitesi Analizi")
        print("6 - Model Eğit (Logistic Reg, Decision Tree, Random Forest)")
        print("7 - Model Sonuçlarını Karşılaştır")
        print("8 - Raporları Kaydet (TXT & PDF)")
        print("9 - Confusion Matrix Grafiğini Kaydet")
        print("10 - Yeni E-Posta İçin Canlı Tahmin")
        print("0 - Çıkış")

        secim = input("\nSeçiminiz (0-10): ").strip()

        if secim == "1":
            load_csv(state)
        elif secim == "2":
            clean_data(state)
        elif secim == "3":
            preview_data(state)
        elif secim == "4":
            analyze_quality(state)
        elif secim == "6":
            train_models(state)
        elif secim == "7":
            compare_results(state)
        elif secim == "8":
            generate_reports(state)
        elif secim == "9":
            save_confusion_matrix(state)
        elif secim == "10":
            predict_new(state)
        elif secim == "0":
            print("\nProgram başarıyla kapatıldı. İyi günler!")
            break
        else:
            print(Fore.RED + "\nGeçersiz seçim! Lütfen menüdeki sayılardan birini girin." + Style.RESET_ALL)


if __name__ == "__main__":
    main()
