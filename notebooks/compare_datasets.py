import os
from tqdm import tqdm
from dotenv import load_dotenv
import numpy as np
import ot
import s3fs
from qdrant_client import QdrantClient
import pandas as pd

load_dotenv(override=True)
# Configuration Client
QDRANT_URL = "http://qdrant:6333"
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=30
)
COLLECTION_A = "naive_synth"
COLLECTION_B = "retext_exhaustive"
COLLECTION_C = "test_prod"
OUTPUT_FILE = "s3://mateom/graal/compare_datasets_test.parquet"
VECTOR_NAME = ""


def get_embeddings_for_code(collection_name: str, code_value: str, limit: int = 1000):
    """
    Récupère les embeddings pour un code donné dans une collection.
    """
    vectors = []
    scroll_filter = {
        "must": [{"key": "code", "match": {"value": code_value}}]
    }

    res, next_page = client.scroll(
        collection_name=collection_name,
        scroll_filter=scroll_filter,
        with_vectors=True,
        limit=limit
    )

    for point in res:
        # Ajuster selon si vos vecteurs sont nommés ou non
        vec = point.vector if VECTOR_NAME == "" else point.vector[VECTOR_NAME]
        if vec is not None:
            vectors.append(vec)

    return np.array(vectors)


def calculate_wasserstein_distance(xs, xt):
    """
    Calcule l'approximation de la distance de Wasserstein entre deux nuages de points.
    """
    n_s, d_s = xs.shape
    n_t, d_t = xt.shape

    # Cas critique : Un des groupes n'a pas de données du tout
    if n_s == 0 or n_t == 0:
        return float('inf')

    # STRATÉGIE POUR LES CODES RARES :
    # Si on a très peu de points, l'Optimal Transport exact est bruité mais reste calculable.
    # Si le nombre de points est trop faible (ex: < 5), la distance entre les moyennes (centroïdes)
    # est le proxy le plus robuste pour éviter de surestimer la distance à cause du manque de données.
    if n_s < 5 or n_t < 5:
        mean_s = np.mean(xs, axis=0)
        mean_t = np.mean(xt, axis=0)
        return float(np.linalg.norm(mean_s - mean_t))

    MAX_POINTS = 2000
    if n_s > MAX_POINTS:
        xs = xs[np.random.choice(n_s, MAX_POINTS, replace=False)]
        n_s = MAX_POINTS
    if n_t > MAX_POINTS:
        xt = xt[np.random.choice(n_t, MAX_POINTS, replace=False)]
        n_t = MAX_POINTS

    # Matrice de coût au carré (Distance Euclidienne entre chaque paire de points)
    M = ot.dist(xs, xt, metric='euclidean')

    # Distributions uniformes sur les deux ensembles de points
    a, b = np.ones((n_s,)) / n_s, np.ones((n_t,)) / n_t

    # Calcul de la distance d'Optimal Transport (Earth Mover's Distance)
    # ot.emd2 renvoie le coût total (Wasserstein W1 ou W2 selon la puissance de la métrique)
    wasserstein_dist = ot.emd2(a, b, M)

    return float(np.sqrt(wasserstein_dist)) # Racine carrée si on veut s'apparenter à W2


def compare_distribution_pipeline(target_code: str):
    vecs_A = get_embeddings_for_code(COLLECTION_A, target_code)
    vecs_B = get_embeddings_for_code(COLLECTION_B, target_code)
    vecs_C = get_embeddings_for_code(COLLECTION_C, target_code.replace(".", ""))

    if len(vecs_A) == 0 or len(vecs_B) == 0 or len(vecs_C) == 0:
        print(f"Code {target_code} -> A: {len(vecs_A)}, B: {len(vecs_B)}, C: {len(vecs_C)}")
        return {}

    distance_AC = calculate_wasserstein_distance(vecs_A, vecs_C)
    distance_BC = calculate_wasserstein_distance(vecs_B, vecs_C)
    distance_AB = calculate_wasserstein_distance(vecs_A, vecs_B)

    return {
        target_code: {
            "count_A": len(vecs_A),
            "count_B": len(vecs_B),
            "count_C": len(vecs_C),
            "wasserstein_distance_AC": distance_AC,
            "wasserstein_distance_BC": distance_BC,
            "wasserstein_distance_AB": distance_AB
        }
    }


if __name__ == "__main__":
    fs = s3fs.S3FileSystem(
        endpoint_url="https://minio.lab.sspcloud.fr",
        client_kwargs={"region_name": "us-east-1"},
    )

    with fs.open("s3://mateom/graal/distribution_zones.npz", "rb") as f:
        with np.load(f, allow_pickle=True) as data:
            head = data["head"]
            body = data["body"]

    metrics = dict()

    for code in tqdm(head):
        metrics.update(compare_distribution_pipeline(code))
    for code in tqdm(body):
        metrics.update(compare_distribution_pipeline(code))

    df_metrics = pd.DataFrame.from_dict(metrics, orient="index")

    df_metrics.to_parquet(OUTPUT_FILE, filesystem=fs)
