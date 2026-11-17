import hashlib
import json
import re

from sklearn.feature_extraction.text import TfidfVectorizer

from .data import normalize, policy_documents, read_products

ALIASES = {
    "cash on delivery": "cod", "ka refund": "refund", "wapas": "return",
    "रिफंड": "refund", "वापस": "return", "वापसी": "return", "कैश ऑन डिलीवरी": "cod",
    "डिलीवरी": "delivery", "भुगतान": "payment", "वारंटी": "warranty",
    "इलेक्ट्रॉनिक्स": "electronics", "जूते": "footwear", "कपड़े": "apparel",
    "kitne din": "timeline", "kitna": "how much", "kab": "when",
    "paise": "money", "wireless earbuds": "bluetooth earbuds",
    "earphones": "bluetooth earbuds", "headphones": "bluetooth earbuds",
    "smartwatch": "smart watch", "waterproof": "water resistance",
    "rtn": "return", "shipping fee": "shipping charges",
}

def search_text(text):
    text = text.casefold()
    for source, target in ALIASES.items():
        text = re.sub(r"(?<!\w)" + re.escape(source) + r"(?!\w)", target, text)
    return text


def product_doc(product):
    fields = []
    for key in ("name", "brand", "category", "rating", "description"):
        if product[key] is not None:
            fields.append(f"{key}: {product[key]}")
    if product["price_paise"] is not None:
        fields.append(f"price: INR {product['price_paise'] / 100:.2f}")
    return {"id": product["sku"], "kind": "product", "title": product["name"] or product["sku"],
            "file": "urbancart_product_catalog_raw.csv", "rows": product["source_rows"],
            "text": "; ".join(fields)}


def render_doc(doc):
    return search_text(" ".join(str(doc.get(key, "")) for key in ("id", "title", "heading", "text")))


class Retrieval:
    def __init__(self, settings, provider=None):
        self.settings = settings
        self.provider = provider
        self.products = {p["sku"]: p for p in read_products()}
        self.policies = policy_documents()
        self.docs = {"product": [product_doc(p) for p in self.products.values()], "policy": self.policies}
        self.by_id = {doc["id"]: doc for docs in self.docs.values() for doc in docs}
        self.lexical = {}
        for kind, docs in self.docs.items():
            model = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
            matrix = model.fit_transform([render_doc(doc) for doc in docs])
            self.lexical[kind] = model, matrix
        self.vector = {}
        self.names = sorted({p["name"] for p in self.products.values() if p["name"]}, key=lambda name: (-len(name), name))

    def index(self):
        import chromadb

        root = self.settings.runtime / "chroma"
        root.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(root))
        usage = []
        for kind, docs in self.docs.items():
            payload = [render_doc(doc) for doc in docs]
            fingerprint = hashlib.sha256(json.dumps([self.settings.embedding_model, payload]).encode()).hexdigest()
            collection = client.get_or_create_collection(
                f"{kind}-{fingerprint[:20]}", embedding_function=None,
                metadata={"embedding_model": self.settings.embedding_model, "fingerprint": fingerprint},
            )
            existing_count = collection.count()
            for start in range(existing_count, len(docs), 64):
                batch = docs[start:start + 64]
                texts = payload[start:start + len(batch)]
                vectors = self.provider.embed(texts, usage)
                collection.upsert(ids=[doc["id"] for doc in batch], documents=texts, embeddings=vectors,
                                  metadatas=[{"file": doc["file"], "kind": kind} for doc in batch])
            self.vector[kind] = collection
        manifest = self.settings.runtime / "index_manifest.json"
        previous = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
        manifest.write_text(json.dumps({
            "embedding_model": self.settings.embedding_model,
            "collections": {key: {"name": col.name, "count": col.count(), "dimension":
                len(col.get(limit=1, include=["embeddings"])["embeddings"][0])}
                for key, col in self.vector.items()},
            "api_calls": previous.get("api_calls", []) + usage,
        }, indent=2), encoding="utf-8")

    def search(self, query, kind, limit=5, usage=None):
        if self.vector:
            vector = self.provider.embed([search_text(query)], usage if usage is not None else [])[0]
            result = self.vector[kind].query(query_embeddings=[vector], n_results=limit, include=[])
            return [self.by_id[key] for key in result["ids"][0]]
        model, matrix = self.lexical[kind]
        scores = (matrix @ model.transform([search_text(query)]).T).toarray().ravel()
        return [self.docs[kind][i] for i in scores.argsort()[::-1][:limit] if scores[i] > 0]

    def match_names(self, query):
        query = normalize(search_text(query))
        return [name for name in self.names if f" {normalize(name)} " in f" {query} "]
