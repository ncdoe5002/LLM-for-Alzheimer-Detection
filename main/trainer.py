import os
import torch
from tqdm import tqdm
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType

# -----------------------------
# 1. LOAD DATA (Updated to use 0 and 1)
# -----------------------------
def load_data(path, isControl, maxInputs):
    dat_path = os.path.join(os.getcwd(), path)
    
    # Sequence Classification needs labels (Control or Dementia)
    label_id = 0 if isControl else 1

    patient_samples = []
    for counter in tqdm(range(1, maxInputs + 1), desc=f"Loading Label {label_id}"):
        file_path = os.path.join(dat_path, f"par_{counter}.txt")

        if not os.path.exists(file_path):
            print(f"Missing: {file_path}")
            break

        with open(file_path, "r", encoding="utf-8") as f:
            transcript = f.read()

        patient_samples.append({
            "text": transcript,
            "label": label_id
        })

    return patient_samples

# Load 20 samples of each
control_data = load_data(os.path.join("data", "control"), True, 30)
dementia_data = load_data(os.path.join("data", "dementia"), False, 40)
all_data = control_data + dementia_data

hf_dataset = Dataset.from_list(all_data)

# -----------------------------
# 2. TOKENIZER + MODEL
# -----------------------------
model_name = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load for Sequence Classification
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2, # 0: Control, 1: Dementia
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.problem_type = "single_label_classification"

# -----------------------------
# 3. APPLY LORA (TaskType: SEQ_CLS)
# -----------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj","up_proj", "down_proj", "gate_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_CLS 
)
model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

# -----------------------------
# 4. TOKENIZATION
# -----------------------------
def tokenize_fn(batch):
    tokens = tokenizer(
        batch["text"],
        truncation=True,
        max_length=512
    )
    tokens["labels"] = batch["label"]
    return tokens

# remove_columns=['text'] prevents the "str" error in the trainer
tokenized_dataset = hf_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

# -----------------------------
# 5. TRAINING CONFIG
# -----------------------------
training_args = TrainingArguments(
    output_dir="./alzheimer_classifier_lora",
    num_train_epochs=10,              
    per_device_train_batch_size=4,    
    learning_rate=1e-5,               
    weight_decay=0.05,
    label_smoothing_factor=0.0,       
    bf16=True,
    logging_steps=1,
    report_to="none"
)

# -----------------------------
# 6. TRAINER
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
)

print("Starting LoRA Classification Training...")
trainer.train()

trainer.save_model("./model/alzheimer_classifier")
tokenizer.save_pretrained("./model/alzheimer_classifier")

print("Training complete. Classifier adapter saved.")
