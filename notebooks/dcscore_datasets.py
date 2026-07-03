import os
from tqdm import tqdm
from dotenv import load_dotenv
import numpy as np
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
COLLECTION_C = "original_cleaned"
OUTPUT_FILE = "s3://mateom/graal/dcscore.parquet"
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
        vec = point.vector if VECTOR_NAME == "" else point.vector[VECTOR_NAME]
        if vec is not None:
            vectors.append(vec)

    return np.array(vectors)


def calculate_dcscore_cs(embeddings_arr: np.ndarray, tau: float = 1.0) -> float:
    """
    Calcule le DCScore en utilisant le kernel Cosine Similarity (cs).
    Implémentation NumPy basée sur le code du papier original.
    """
    n_samples = embeddings_arr.shape[0]
    if n_samples == 0:
        return 0.0
    if n_samples == 1:
        return 1.0

    # 1. Produit matriciel pour la similarité cosinus 
    # (Note : s'assurer en amont que les embeddings sont normalisés L2 si requis par votre modèle)
    sim_product = (embeddings_arr @ embeddings_arr.T) / tau
    
    # 2. Softmax stable par ligne (dim=-1)
    max_sim = np.max(sim_product, axis=-1, keepdims=True)
    exp_sim = np.exp(sim_product - max_sim)
    sim_probs = exp_sim / np.sum(exp_sim, axis=-1, keepdims=True)
    
    # 3. Somme de la diagonale (Trace)
    diversity = np.sum(np.diag(sim_probs))
    
    return float(diversity)


def process_code_diversity_pipeline(target_code: str):
    vecs_A = get_embeddings_for_code(COLLECTION_A, target_code)
    vecs_B = get_embeddings_for_code(COLLECTION_B, target_code)
    vecs_C = get_embeddings_for_code(COLLECTION_C, target_code)

    # Calcul des scores (renvoie 0.0 si la collection est vide pour ce code)
    dcscore_A = calculate_dcscore_cs(vecs_A, tau=1.0)
    dcscore_B = calculate_dcscore_cs(vecs_B, tau=1.0)
    dcscore_C = calculate_dcscore_cs(vecs_C, tau=1.0)

    return {
        target_code: {
            "count_A": len(vecs_A),
            "dcscore_A": dcscore_A,
            "count_B": len(vecs_B),
            "dcscore_B": dcscore_B,
            "count_C": len(vecs_C),
            "dcscore_C": dcscore_C
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

    print("Traitement de la zone 'head'...")
    for code in tqdm(head):
        metrics.update(process_code_diversity_pipeline(code))
        
    print("Traitement de la zone 'body'...")
    for code in tqdm(body):
        metrics.update(process_code_diversity_pipeline(code))

    df_metrics = pd.DataFrame.from_dict(metrics, orient="index")

    # Sauvegarde du résultat final au format parquet
    df_metrics.to_parquet(OUTPUT_FILE, filesystem=fs)
    print(f"Pipeline terminé. Fichier enregistré sur {OUTPUT_FILE}")