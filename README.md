# Kawan Kampus Recommendation API

Kawan Kampus Recommendation API adalah layanan REST API berbasis FastAPI yang menyediakan rekomendasi dan pencarian tempat di sekitar kampus berdasarkan kebutuhan mahasiswa.

API ini menggunakan model Machine Learning berbasis TensorFlow untuk menghasilkan rekomendasi tempat yang relevan, serta menggunakan TF-IDF dan Cosine Similarity untuk mendukung fitur pencarian tempat berdasarkan kata kunci.

## Fitur Utama

* Rekomendasi tempat berdasarkan kampus pengguna
* Filter berdasarkan kategori kebutuhan
* Filter berdasarkan kategori jarak
* Prediksi skor rekomendasi menggunakan model Machine Learning
* Pencarian tempat menggunakan TF-IDF dan Cosine Similarity
* Menampilkan informasi tempat, rating, jumlah ulasan, dan tautan Google Maps
* REST API berbasis FastAPI
* Dokumentasi API otomatis menggunakan Swagger UI

## Teknologi yang Digunakan

* Python 3.11
* FastAPI
* TensorFlow
* Scikit-Learn
* Pandas
* NumPy
* Joblib
* Uvicorn
* TF-IDF
* Cosine Similarity

## Struktur Proyek

```text
Kawan-Kampus/
│
├── app.py
├── recommender.py
├── requirements.txt
├── Dockerfile
│
├── models/
│   ├── recommender_system.keras
│   ├── kategori_encoder.pkl
│   ├── kampus_encoder.pkl
│   ├── jarak_encoder.pkl
│   ├── scaler.pkl
│   ├── tfidf.pkl
│   ├── search_tfidf.pkl
│   └── search_matrix.pkl
│
├── data/
│   └── kawankampus_master_dataset.csv
│
└── README.md
```

## Menjalankan API Secara Lokal

### 1. Clone Repository

```bash
git clone https://github.com/username/kawan-kampus-api.git
cd kawan-kampus-api
```

### 2. Membuat Virtual Environment

```bash
python -m venv venv
```

### 3. Aktivasi Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/MacOS:

```bash
source venv/bin/activate
```

### 4. Install Dependency

```bash
pip install -r requirements.txt
```

### 5. Menjalankan API

```bash
python -m uvicorn app:app --reload
```

API akan berjalan pada:

```text
http://127.0.0.1:8000
```

Swagger Documentation:

```text
http://127.0.0.1:8000/docs
```

## Endpoint API

### GET /

Memeriksa status API.

#### Response

```json
{
  "message": "Kawan Kampus Recommendation API is running"
}
```

---

### POST /recommend

Menghasilkan rekomendasi tempat berdasarkan kampus, kategori, dan kategori jarak.

#### Request Body

```json
{
  "kampus": "Universitas Multi Data Palembang",
  "kategori": "Cafe",
  "kategori_jarak": "Jalan Kaki",
  "top_n": 10
}
```

#### Response

```json
[
  {
    "Nama_Tempat": "De'warkop",
    "Kategori_Awal": "Cafe",
    "Kategori_Jarak": "Jalan Kaki",
    "Kampus": "Universitas Multi Data Palembang",
    "Rating": 4.7,
    "Total_Reviews": 150,
    "Jarak_KM": 0.4,
    "Google_Maps_Link": "https://maps.google.com/...",
    "recommendation_score": 0.95
  }
]
```

---

### POST /recommend/all

Menghasilkan rekomendasi untuk seluruh kategori tempat yang tersedia pada kampus yang dipilih.

#### Request Body

```json
{
  "kampus": "Universitas Multi Data Palembang",
  "kategori_jarak": "Jalan Kaki",
  "top_n": 5
}
```

#### Response

```json
{
  "Cafe": [
    {
      "Nama_Tempat": "De'warkop",
      "recommendation_score": 0.95
    }
  ],
  "Fotokopi": [
    {
      "Nama_Tempat": "XYZ Printing",
      "recommendation_score": 0.92
    }
  ]
}
```

---

### POST /search

Melakukan pencarian tempat menggunakan TF-IDF dan Cosine Similarity.

Pencarian dibatasi pada kampus yang dipilih pengguna sehingga hasil yang ditampilkan tetap relevan dengan area kampus tersebut.

#### Request Body

```json
{
  "kampus": "Universitas Multi Data Palembang",
  "query": "warkop",
  "top_n": 10
}
```

#### Response Berhasil

```json
{
  "message": "Hasil pencarian ditemukan.",
  "data": [
    {
      "Nama_Tempat": "De'warkop",
      "Kategori_Awal": "Cafe",
      "Kampus": "Universitas Multi Data Palembang",
      "Rating": 4.7,
      "Total_Reviews": 150,
      "Jarak_KM": 0.4,
      "Google_Maps_Link": "https://maps.google.com/...",
      "similarity": 0.84
    }
  ]
}
```

#### Response Jika Tidak Ditemukan

```json
{
  "message": "Tempat yang dicari tidak ditemukan."
}
```

## Cara Kerja Sistem

### Recommendation Engine

Digunakan pada endpoint:

```text
POST /recommend
POST /recommend/all
```

Proses:

1. Memfilter data berdasarkan kampus.
2. Memfilter kategori dan kategori jarak.
3. Membentuk fitur numerik dan TF-IDF.
4. Menghasilkan recommendation score menggunakan model TensorFlow.
5. Mengurutkan hasil berdasarkan recommendation score tertinggi.

### Search Engine

Digunakan pada endpoint:

```text
POST /search
```

Proses:

1. Query pengguna diubah menjadi representasi TF-IDF.
2. Sistem menghitung Cosine Similarity terhadap data tempat.
3. Hasil difilter berdasarkan kampus yang dipilih pengguna.
4. Tempat dengan similarity tertinggi dikembalikan sebagai hasil pencarian.

## Deployment

API dirancang untuk dapat dideploy menggunakan:

* Google Cloud Run
* Docker
* Railway
* Render
* VPS berbasis Linux

## Lisensi

Proyek ini dibuat untuk kebutuhan Capstone Project Kawan Kampus dan tujuan pembelajaran Machine Learning serta Backend Development.
