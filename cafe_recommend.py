import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)
pd.set_option("display.width", 500)
# Veri setini yükle
df = pd.read_excel("datasets/mekan.xlsx", sheet_name="Mekan")
# Gerekli dönüşümleri yap
df["Yaş"] = df["Yaş"].astype(str)
df["Fiyat"] = df["Fiyat"].astype(str)
df["Desc"] = df["Tür"] + ", " + df["Fiyat"] + ", " + df["DiyetGer"]

# Fiyat bilgilerini sayısal değerlere dönüştürmek için bir fonksiyon tanımlayın
def fiyat_to_numeric(fiyat):
    try:
        return float(fiyat)
    except ValueError:
        return np.nan

df["FiyatNumeric"] = df["Fiyat"].apply(fiyat_to_numeric)

# Kullanıcıdan giriş al
tur_secim = input("Mekan Türü Seçiniz: ").strip()
butce_secim = float(input("Bütçe Giriniz: ").strip())

# İlk 30 satırı filtrele
df = df.iloc[:30]

# Bütçeye uygun mekanları filtrele
df_filtered = df[df["FiyatNumeric"] <= butce_secim]

# Eğer bütçeye uygun mekan yoksa, uygun bir mesaj gösterin
if df_filtered.empty:
    print("Belirtilen bütçeye uygun mekan bulunamadı.")
else:
    # Kullanıcı seçimini birleştir
    secim = tur_secim + ", " + str(butce_secim)

    # TF-IDF matrisini oluştur
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_filtered['Desc'])

    # Kullanıcı seçimini vektörize et
    secim_tfidf = tfidf.transform([secim])

    # Cosine similarity hesapla
    cosine_sim = linear_kernel(secim_tfidf, tfidf_matrix)

    # Benzerlik skorlarını al
    sim_scores = list(enumerate(cosine_sim[0]))

    # Benzerlik skorlarına göre sıralama yap
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # En benzer mekanları al
    top_matches = sim_scores[:5]

    # Kullanıcıya önerilecek mekanları yazdır
    print("İşte önerdiğimiz mekanlar:")
    for match in top_matches:
        index = match[0]
        mekan_bilgileri = df_filtered.iloc[index]
        print("\nMekan Bilgileri:")
        for column_name, value in mekan_bilgileri.items():
            print(f"{column_name}: {value}", end="\t")
        print("\n")
