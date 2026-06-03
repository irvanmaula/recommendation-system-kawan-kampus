import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras import layers
from sklearn.metrics.pairwise import cosine_similarity
# ==================================
# LOAD DATA
# ==================================

data = pd.read_csv(
    "data/kawankampus_master_dataset.csv"
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

search_tfidf = joblib.load(
    'models/search_tfidf.pkl'
)

search_matrix = joblib.load(
    'models/search_matrix.pkl'
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


def search_similar_places(
    kampus,
    query,
    top_n=10
):

    query_vector = (
        search_tfidf.transform(
            [query]
        )
    )

    similarities = (
        cosine_similarity(
            query_vector,
            search_matrix
        )
        .flatten()
    )

    result = data.copy()

    result["similarity"] = similarities

    result = result[
        result["Kampus"]
        .str.contains(
            kampus,
            case=False,
            na=False
        )
    ]

    result = result.sort_values(
        by="similarity",
        ascending=False
    )

    if result.empty:
        return pd.DataFrame()

    if result["similarity"].max() <= 0:
        return pd.DataFrame()

    return result[[

        "Nama_Tempat",

        "Kategori_Awal",

        "Kampus",

        "Rating",

        "Total_Reviews",

        "Jarak_KM",

        "Google_Maps_Link",

        "similarity"

    ]].head(top_n)

def recommend_places(
    kampus,
    kategori,
    kategori_jarak,
    top_n=10,
    use_fallback=True
):

    filtered = data[

        data['Kampus']
        .str.contains(
            kampus,
            case=False,
            na=False
        )

        &

        (
            data['Kategori_Awal']
            == kategori
        )

        &

        data['Kategori_Jarak']
        .str.contains(
            kategori_jarak,
            case=False,
            na=False
        )

    ].copy()

    # ==================================
    # FALLBACK
    # ==================================

    if filtered.empty:
        return pd.DataFrame()

    # ==================================
    # LANJUTKAN MODEL REKOMENDASI
    # ==================================

    # TF-IDF
    filtered['combined_features'] = (
        filtered['Kategori_Awal'].astype(str)
        + ' ' +
        filtered['Kategori_Jarak'].astype(str)
        + ' ' +
        filtered['Kampus'].astype(str)
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

def recommend_all_categories(
    kampus,
    kategori_jarak,
    top_n=10
):

    results = {}

    categories = sorted(
        data["Kategori_Awal"]
        .dropna()
        .unique()
    )

    for kategori in categories:

        hasil = recommend_places(
            kampus=kampus,
            kategori=kategori,
            kategori_jarak=kategori_jarak,
            top_n=top_n,
            use_fallback=False
        )

        if isinstance(
            hasil,
            dict
        ):
            continue

        if not hasil.empty:
            results[kategori] = hasil.to_dict(
                orient="records"
            )
    return results