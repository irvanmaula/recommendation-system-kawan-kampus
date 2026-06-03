from fastapi import FastAPI
from fastapi import Form
from pydantic import BaseModel


from recommender import (
    recommend_places,
    recommend_all_categories,
    search_similar_places
)

app = FastAPI(
    title="Kawan Kampus Recommendation API"
)

# ==================================
# REQUEST SCHEMA
# ==================================

class RecommendationRequest(BaseModel):

    kampus: str

    kategori: str

    kategori_jarak: str

    top_n: int = 10


class RecommendationAllRequest(
    BaseModel
):

    kampus: str

    kategori_jarak: str

    top_n: int = 10

class SearchRequest(BaseModel):
    kampus: str
    query: str
    top_n: int = 10

# ==================================
# HEALTH CHECK
# ==================================

@app.get("/")
def home():

    return {
        "message":
        "Kawan Kampus Recommendation API"
    }


# ==================================
# RECOMMENDATION
# ==================================

@app.post("/recommend")
def recommend(

    request:
    RecommendationRequest

):

    hasil = recommend_places(

        kampus=request.kampus,

        kategori=request.kategori,

        kategori_jarak=request.kategori_jarak,

        top_n=request.top_n

    )

    if isinstance(hasil, dict):
        return hasil

    return hasil.to_dict(
        orient="records"
    )

# ==================================
# RECOMMENDATION ALL
# ==================================

@app.post("/recommend/all")
def recommend_all(

    request:
    RecommendationAllRequest

):

    hasil = recommend_all_categories(

        kampus=request.kampus,

        kategori_jarak=request.kategori_jarak,

        top_n=request.top_n

    )

    return hasil

# =================================
# SEARCH PLACE
# =================================

@app.post("/search")

def search(
    request: SearchRequest
):
    result = search_similar_places(
        kampus=request.kampus,
        query=request.query,
        top_n=request.top_n
    )

    if result.empty:
        return {
            "message":
            "Tempat yang dicari tidak ditemukan."
        }

    return {
        "message":
        "Hasil pencarian ditemukan.",
        "data":
        result.to_dict(
            orient="records"
        )
    }