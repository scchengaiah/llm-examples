from pathlib import Path
from qdrant_client import QdrantClient
import json

from code_search.config import QDRANT_URL, QDRANT_API_KEY, DATA_DIR, QDRANT_FILE_COLLECTION_NAME
from code_search.index.utils import measure_execution_time

@measure_execution_time
def encode_and_upload():
    qdrant_client = QdrantClient(
        QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )

    collection_name = QDRANT_FILE_COLLECTION_NAME
    input_file = Path(DATA_DIR) / "py_files.json"

    if not input_file.exists():
        raise RuntimeError(f"File {input_file} does not exist. Skipping")

    payload = []
    with open(input_file, 'r') as json_file:
        data = json.load(json_file)
        payload = data

    print(f"Recreating the collection {collection_name}")
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config={}
    )

    print(f"Storing data in the collection {collection_name}")
    print(f"Payload length: {len(payload)}")
    qdrant_client.upload_collection(
        collection_name=collection_name,
        payload=payload,
        vectors=[{}] * len(payload),
        ids=None,
        batch_size=20
    )

if __name__ == '__main__':
    encode_and_upload()
    
