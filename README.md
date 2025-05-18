## Recommender System for Movie Recommendations
Detta är ett enkelt rekommendationssystem för filmer som använder SVD (Singular Value Decomposition) och TF-IDF (Term Frequency-Inverse Document Frequency) för att ge rekommendationer baserat på användarens betyg och filmbeskrivningar.

# Datasetet som används:
- https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=movies_metadata.csv

Lägger i mappen `data/` i projektmappen. (Du kan behöva skapa en mapp som heter `data/` om den inte redan finns.)

## How to use
1. Klona eller ladda ner detta repository.
2. Navigera till projektmappen i terminalen.
3. Skapa och aktivera en virtuell miljö (valfritt)
4. Installera nödvändiga paket.
5. Lägg till dina egna dataset i `data/`-mappen. 'movies_metadata.csv' och 'ratings.csv eller ratings_small.csv' är exempel på dataset som kan användas.
6. Följ instruktionerna i terminalen för att ge betyg på filmer och få rekommendationer. Ändra det sökta filmens titel i `movie_rec_system.py` för att få rekommendationer för en specifik film i main metoden längst ner söka efter en film i `movie_rec_system.py`-filen. Exempel:

```python
search_title = "The Matrix Reloaded"
```

7. Kör `movie_rec_system.py` för att starta programmet.


## Installation
### 1. Skapa och aktivera en virtuell miljö (valfritt men rekommenderat)

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### . Installera paketen

När du är i projektmappen, kör:

```bash
pip install -r requirements.txt
```

Detta installerar:

* `pandas` för datahantering
* `numpy` för numeriska beräkningar
* `scikit-learn` för maskininlärning (SVD, TF-IDF, cosine similarity)


