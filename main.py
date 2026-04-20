import pandas as pd
from search_context import build_index, get_best_context
from model import load_qa_model, get_answer

# Load dataset
df = pd.read_csv("data/train.csv")
df = df.head(3000)

# Build index
print("🔄 Building search index...")
contexts = df['context'].tolist()
index, _ = build_index(contexts)

# Load model
print("🤖 Loading QA model...")
tokenizer, model = load_qa_model()

print("🚀 Smart QA System Ready! Type 'exit' to stop.")

while True:
    question = input("\nAsk your question: ")

    if question.lower() == "exit":
        print("👋 Exiting...")
        break

    context = get_best_context(question, df, index)

    answer = get_answer(question, context, tokenizer, model)

    print("\n💬 Answer:", answer)