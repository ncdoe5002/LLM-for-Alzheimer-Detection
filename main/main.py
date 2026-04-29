import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

model_id = "meta-llama/Llama-3.2-1B-Instruct"
adapter_path = "./model/alzheimer_classifier"

tokenizer = AutoTokenizer.from_pretrained(model_id)

base_model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=2,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()


def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()

    labels = ["Control", "Dementia"]
    confidence = probs[0][pred].item() * 100

    return pred, labels[pred], confidence


def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")

    label = int(lines[0])          # first line
    text = "\n".join(lines[1:])    # rest is transcript

    return label, text

def do_test(folder):

    files = []
    for f in os.listdir(folder):
        if f.endswith(".txt"):
            files.append(f)

    correct = 0
    total = 0

    print("\nFolder:", folder)
    print("=" * 40)

    for file in files:

        path = folder + "/" + file

        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().strip().split("\n")

        true_label = int(lines[0])
        text = "\n".join(lines[1:]).strip()

        pred, label_name, conf = predict(text)

        if pred == true_label:
            correct += 1

        total += 1

        print(file)
        print("Pred:", label_name, "| Conf:", round(conf, 2))
        print("True:", true_label)
        print("-" * 30)

    if total == 0:
        print("No files found!")
    else:
        print("\nAccuracy:", round(correct / total * 100, 2), "%")

do_test("./data/test")
