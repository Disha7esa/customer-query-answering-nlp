from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

def load_qa_model():
    model_name = "distilbert-base-cased-distilled-squad"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    return tokenizer, model


def get_answer(question, context, tokenizer, model):
    inputs = tokenizer(question, context, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits) + 1

    answer_tokens = inputs["input_ids"][0][start_idx:end_idx]
    answer = tokenizer.decode(answer_tokens)

    return answer