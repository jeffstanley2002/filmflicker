import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def synthetic_movies():
    return pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4, 5],
            "title": [
                "Action Hero (2001)",
                "Rom Com Classic (1999)",
                "Sci-Fi Epic (2010)",
                "No Year Movie",
                "Indie Drama (2015)",
            ],
            "genres": ["Action|Thriller", "Comedy|Romance", "Sci-Fi|Action", "Drama", "Drama|Romance"],
        }
    )


@pytest.fixture
def synthetic_movies_featured(synthetic_movies):
    df = synthetic_movies.copy()
    df["year"] = df["title"].apply(
        lambda t: int(t[-5:-1]) if t.endswith(")") else None
    )
    df["genre_list"] = df["genres"].apply(lambda g: g.split("|"))
    return df


@pytest.fixture
def synthetic_ratings():
    # 3 users rating overlapping subsets of the 5 synthetic movies.
    return pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 3, 3, 3, 3],
            "movieId": [1, 2, 3, 1, 3, 1, 2, 4, 5],
            "rating": [5.0, 3.0, 4.0, 4.0, 5.0, 2.0, 4.5, 3.5, 5.0],
            "timestamp": [0] * 9,
        }
    )
