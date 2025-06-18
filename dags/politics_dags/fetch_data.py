import re

import pendulum
from airflow import DAG
from airflow.providers.weaviate.hooks.weaviate import WeaviateHook
from airflow.sdk import Asset, chain, task

COLLECTION_NAME = "Politics"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RSS_FEED_URL = "https://feeds.npr.org/1014/rss.xml"

default_args = {
    "retries": 2,
}

with DAG(
    dag_id="political_news_scraper",
    default_args=default_args,
    start_date=pendulum.today('UTC').add(days=-1),
    schedule="@daily",
    catchup=False,
    tags=["politics", "news", "scraping"],
) as dag:

    @task
    def fetch_article_urls():
        import feedparser

        feed = feedparser.parse(RSS_FEED_URL)
        return [entry.link for entry in feed.entries]

    @task
    def scrape_article(url: str):
        import requests
        from bs4 import BeautifulSoup
        import logging

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            title_tag = soup.find("title")
            title = title_tag.text.strip() if title_tag else "No title"

            paragraphs = soup.find_all("p")
            text = "\n".join(p.get_text().strip() for p in paragraphs if p.get_text().strip())

            if not text:
                logging.warning(f"No content extracted from {url}")
                return {"url": url, "error": "No content found"}

            return {"url": url, "title": title, "content": text}

        except requests.exceptions.RequestException as e:
            # Network/HTTP errors - log and continue
            logging.error(f"Request failed for {url}: {e}")
            return {"url": url, "error": f"Request error: {str(e)}"}
        except Exception as e:
            # Parsing errors - log and continue
            logging.error(f"Parsing failed for {url}: {e}")
            return {"url": url, "error": f"Parsing error: {str(e)}"}

    @task
    def clean_article(article: dict) -> dict:
        cleaned = re.sub(r"\s+", " ", article.get("content", "")).strip()
        article["content"] = cleaned
        return article
    
    @task(outlets=[Asset("politics_vector_data")], trigger_rule="all_done")
    def index_to_weaviate(articles: list[dict]) -> str:
        from llama_index.core import Document, StorageContext, VectorStoreIndex
        from llama_index.embeddings.fastembed import FastEmbedEmbedding
        from llama_index.vector_stores.weaviate import WeaviateVectorStore

        documents = [Document(text=article["content"], metadata={"url": article["url"], "title": article.get("title", "")}) for article in articles if "content" in article and article["content"]]

        hook = WeaviateHook("my_weaviate_conn")
        client = hook.get_conn()

        vector_store = WeaviateVectorStore(weaviate_client=client, index_name=COLLECTION_NAME)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=FastEmbedEmbedding(EMBEDDING_MODEL_NAME)
        )
        return "Indexing complete"

    _fetch_article_urls = fetch_article_urls()
    _scrape_article = scrape_article.expand(url=_fetch_article_urls)
    _clean_article = clean_article.expand(article=_scrape_article)
    _index_to_weaviate = index_to_weaviate(_clean_article)
    chain(_fetch_article_urls, _scrape_article, _clean_article, _index_to_weaviate)
