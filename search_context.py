from sentence_transformers import SentenceTransformer
import faiss

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def build_index(texts):
    embeddings = model.encode(texts, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

def get_best_context(query, df, index):
    query_vec = model.encode([query])
    D, I = index.search(query_vec, 1)
    return df.iloc[I[0][0]]['context']