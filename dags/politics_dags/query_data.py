from airflow import DAG
from airflow.sdk import task, Asset

COLLECTION_NAME = "Politics"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

with DAG(
    dag_id="political_news_searcher",
    schedule=[Asset("politics_vector_data")],
    params={"query_str": "Donald Trump"},
) as dag:

    @task
    def inspect_weaviate_collection():
        import logging
        from airflow.providers.weaviate.hooks.weaviate import WeaviateHook

        hook = WeaviateHook("my_weaviate_conn")
        client = hook.get_conn()
        collection = client.collections.get(COLLECTION_NAME)
        objs = collection.query.fetch_objects()
        logging.info("%d objects found in the collection.", len(objs.objects))
        client.close()

    @task
    def search_vector_db(**context):
        import logging
        from airflow.providers.weaviate.hooks.weaviate import WeaviateHook
        from fastembed import TextEmbedding

        query_str = context["params"]["query_str"]

        hook = WeaviateHook("my_weaviate_conn")
        client = hook.get_conn()

        embedding_model = TextEmbedding(EMBEDDING_MODEL_NAME)
        collection = client.collections.get(COLLECTION_NAME)

        query_emb = list(embedding_model.embed([query_str]))[0]

        results = collection.query.near_vector(
            near_vector=query_emb,
            limit=1,
        )
        for result in results.objects:
            logging.info("Something interesting for you:")
            logging.info("title: %s", result.properties["title"])
            logging.info("url: %s", result.properties["url"])
            logging.info("text: %s", result.properties["text"])
        
        client.close()

    _inspect_weaviate_collection = inspect_weaviate_collection()
    _search_vector_db = search_vector_db()