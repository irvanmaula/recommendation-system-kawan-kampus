import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras import layers

# ==================================
# LOAD DATA
# ==================================

data = pd.read_csv(
    "data/cleaned_places.csv"
)

class FeatureInteractionLayer(layers.Layer):

    def __init__(self, **kwargs):

        super(FeatureInteractionLayer, self).__init__(**kwargs)

    def call(self, inputs):

        x1, x2 = inputs

        return x1 * x2

    def get_config(self):

        config = super(
            FeatureInteractionLayer,
            self
        ).get_config()

        return config

class CustomHuberLoss(tf.keras.losses.Loss):

    def __init__(self, delta=1.0, **kwargs):

        super(CustomHuberLoss, self).__init__(**kwargs)

        self.delta = delta

    def call(self, y_true, y_pred):

        error = y_true - y_pred

        condition = tf.abs(error) <= self.delta

        small_error = 0.5 * tf.square(error)

        large_error = self.delta * (
            tf.abs(error) - 0.5 * self.delta
        )

        return tf.where(
            condition,
            small_error,
            large_error
        )

    def get_config(self):

        config = super(
            CustomHuberLoss,
            self
        ).get_config()

        config.update({
            "delta": self.delta
        })

        return config
# ==================================
# LOAD MODEL
# ==================================

loaded_model = tf.keras.models.load_model(

    "models/recommender_system.keras",

    custom_objects={

        "FeatureInteractionLayer":
        FeatureInteractionLayer,

        "CustomHuberLoss":
        CustomHuberLoss

    },

    compile=False
)
# ==================================
# LOAD PREPROCESSOR
# ==================================

kategori_encoder = joblib.load(
    "models/kategori_encoder.pkl"
)

kampus_encoder = joblib.load(
    "models/kampus_encoder.pkl"
)

jarak_encoder = joblib.load(
    "models/jarak_encoder.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)

tfidf = joblib.load(
    "models/tfidf.pkl"
)

# ==================================
# FEATURE COLUMNS
# ==================================

feature_columns = (

    [

        "kategori_encoded",

        "kampus_encoded",

        "jarak_encoded",

        "Rating",

        "Total_Reviews",

        "Jarak_KM"

    ]

    +

    list(
        tfidf.get_feature_names_out()
    )

)


kategori_mapping = {
    'Makanan': [
        'Makanan',
        'Makanan Siap Saji',
        'Cafe',
        'Kedai',
        'Kedai Kopi',
        'Pizza',
        'Restoran',
        'Restoran Padang',
        'Toko Es Krim',
        'Warteg'
    ],

    'Belanja': [
        'Minimarket'
    ],

    'Kesehatan': [
        'Apotek',
        'Tempat Fitness'
    ],

    'Transportasi': [
        'Perhentian Bus'
    ],
    'Cetak': [
        'Fotokopi',
        'Print'
    ]
}


def recommend_places(

    kampus,
    kategori,
    kategori_jarak,
    top_n=10

):

    kategori_list = kategori_mapping.get(
        kategori,
        [kategori]
    )

    filtered = data[

        data['Kampus']
        .str.contains(
            kampus,
            case=False,
            na=False
        )

        &

        data['Kategori_Awal']
        .isin(kategori_list)

        &

        data['Kategori_Jarak']
        .str.contains(
            kategori_jarak,
            case=False,
            na=False
        )

    ].copy()

    if filtered.empty:

        return "Tidak ada rekomendasi"

    # ======================
    # TF-IDF
    # ======================

    filtered['combined_features'] = (

        filtered['Kategori_Awal']
        .astype(str)

        + ' '

        +

        filtered['Kategori_Jarak']
        .astype(str)

        + ' '

        +

        filtered['Kampus']
        .astype(str)

    )

    filtered_tfidf = tfidf.transform(

        filtered['combined_features']

    )

    filtered_tfidf_df = pd.DataFrame(

        filtered_tfidf.toarray(),

        columns=tfidf.get_feature_names_out(),

        dtype='float32'

    )

    filtered['kategori_encoded'] = (
    kategori_encoder.transform(
        filtered['Kategori_Awal']
      )
    )

    filtered['kampus_encoded'] = (
    kampus_encoder.transform(
        filtered['Kampus']
      )
    )

    filtered['jarak_encoded'] = (
    jarak_encoder.transform(
        filtered['Kategori_Jarak']
      )
    )

    # ======================
    # NUMERIC
    # ======================

    filtered_numeric = filtered[[

        'kategori_encoded',

        'kampus_encoded',

        'jarak_encoded',

        'Rating',

        'Total_Reviews',

        'Jarak_KM'

    ]].copy()

    filtered_numeric[

        [
            'Rating',
            'Total_Reviews',
            'Jarak_KM'
        ]

    ] = scaler.transform(

        filtered_numeric[

            [
                'Rating',
                'Total_Reviews',
                'Jarak_KM'
            ]

        ]

    )

    # ======================
    # FINAL INPUT
    # ======================

    input_data = pd.concat(

        [

            filtered_numeric
            .reset_index(drop=True),

            filtered_tfidf_df
            .reset_index(drop=True)

        ],

        axis=1

    ).astype("float32")

    input_data = input_data.reindex(

        columns=feature_columns,

        fill_value=0

    )

    predictions = loaded_model.predict(

        input_data,

        verbose=0

    ).flatten()

    filtered[
        "recommendation_score"
    ] = predictions

    filtered = filtered.sort_values(

        by="recommendation_score",

        ascending=False

    )
    

    return filtered[[

    "Nama_Tempat",

    "Kategori_Awal",

    "Kategori_Jarak",

    "Kampus",

    "Rating",

    "Total_Reviews",

    "Jarak_KM",

    "Google_Maps_Link",

    "recommendation_score"

]].head(top_n)