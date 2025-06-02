


# import os
# import sys

# # ----------------- Setup -----------------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(BASE_DIR))

# import torch
# import pandas as pd
# import numpy as np
# from torch import nn
# from datasets import Dataset, DatasetDict
# from transformers import (
#     AutoTokenizer,
#     AutoModelForTokenClassification,
#     TrainingArguments,
#     Trainer,
#     AutoConfig,
#     pipeline,
# )
# from seqeval.metrics import classification_report
# import ast



# DATA_PATH = "data/IT jobs for training.csv"
# SPLIT_DIR = "data/split"
# os.makedirs(SPLIT_DIR, exist_ok=True)

# # ----------------- Annotation -----------------
# def annotate_skills(text, skills):
#     entities = []
#     if pd.isna(skills) or pd.isna(text):
#         return {'text': text, 'entities': []}

#     skill_list = [s.strip() for s in skills.split(',')]
#     text_lower = text.lower()

#     for skill in skill_list:
#         skill_lower = skill.lower()
#         start = 0
#         while True:
#             start = text_lower.find(skill_lower, start)
#             if start == -1:
#                 break
#             end = start + len(skill)
#             entities.append({'start': start, 'end': end, 'label': 'SKILL'})
#             start = end

#     return {'text': text, 'entities': entities}

# # ----------------- Load or Create Splits -----------------
# if os.path.exists(f"{SPLIT_DIR}/train.csv") and os.path.exists(f"{SPLIT_DIR}/test.csv"):
#     print("✅ Using previously saved train/test splits...")
#     train_df = pd.read_csv(f"{SPLIT_DIR}/train.csv")
#     test_df = pd.read_csv(f"{SPLIT_DIR}/test.csv")

#     train_df['entities'] = train_df['entities'].apply(ast.literal_eval)
#     test_df['entities'] = test_df['entities'].apply(ast.literal_eval)
# else:
#     print("🔄 Splitting and saving dataset for the first time...")
#     df = pd.read_csv(DATA_PATH)
#     annotated_data = [annotate_skills(row['job_description'], row['skills']) for _, row in df.iterrows()]
#     full_df = pd.DataFrame(annotated_data)
#     split = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
#     train_df = split.iloc[:int(0.8 * len(split))]
#     test_df = split.iloc[int(0.8 * len(split)):]
#     train_df.to_csv(f"{SPLIT_DIR}/train.csv", index=False)
#     test_df.to_csv(f"{SPLIT_DIR}/test.csv", index=False)

# # Convert to Hugging Face datasets
# dataset = DatasetDict({
#     "train": Dataset.from_pandas(train_df),
#     "test": Dataset.from_pandas(test_df)
# })

# # ----------------- Tokenization -----------------
# model_name = "bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
# label_names = ["O", "B-SKILL", "I-SKILL"]

# def tokenize_and_align_labels(examples):
#     tokenized = tokenizer(examples["text"], truncation=True, padding=True)
#     labels = []

#     for i, entities in enumerate(examples["entities"]):
#         word_ids = tokenized.word_ids(batch_index=i)
#         label_ids = [-100] * len(word_ids)
#         for entity in entities:
#             for idx, word_id in enumerate(word_ids):
#                 if word_id is None:
#                     continue
#                 char_span = tokenized.token_to_chars(i, idx)
#                 if char_span and entity["start"] <= char_span.start < entity["end"]:
#                     label_ids[idx] = 1 if char_span.start == entity["start"] else 2
#         labels.append(label_ids)

#     tokenized["labels"] = labels
#     return tokenized

# tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# # ----------------- Model -----------------
# config = AutoConfig.from_pretrained(
#     model_name,
#     num_labels=3,
#     id2label={0: "O", 1: "B-SKILL", 2: "I-SKILL"},
#     label2id={"O": 0, "B-SKILL": 1, "I-SKILL": 2},
#     hidden_dropout_prob=0.3,
#     attention_probs_dropout_prob=0.3
# )

# model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)

# class CustomTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
#         labels = inputs.pop("labels")
#         outputs = model(**inputs)
#         logits = outputs["logits"]
#         loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.0, 5.0]).to(model.device))
#         loss = loss_fct(logits.view(-1, 3), labels.view(-1))
#         return (loss, outputs) if return_outputs else loss

# def compute_metrics(p):
#     predictions, labels = p
#     predictions = np.argmax(predictions, axis=2)
#     true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
#     true_preds = [[label_names[p] for (p, l) in zip(pred, label) if l != -100] for pred, label in zip(predictions, labels)]
#     report = classification_report(true_labels, true_preds, output_dict=True)
#     return {
#         "precision": report["macro avg"]["precision"],
#         "recall": report["macro avg"]["recall"],
#         "f1": report["macro avg"]["f1-score"],
#         "accuracy": np.mean([float(t == p) for t, p in zip(true_labels, true_preds)]),
#     }

# # ----------------- Training -----------------
# training_args = TrainingArguments(
#     output_dir="./ner_results",
#     eval_strategy="epoch",
#     learning_rate=2e-6,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=10,
#     weight_decay=0.01,
#     logging_dir='./logs',
#     save_strategy="epoch",
#     load_best_model_at_end=True,
#     metric_for_best_model="f1",
#     greater_is_better=True,
# )

# trainer = CustomTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["test"],
#     compute_metrics=compute_metrics,
# )

# print("🚀 Starting training...")
# trainer.train()

# model.save_pretrained("./ner_model")
# tokenizer.save_pretrained("./ner_tokenizer")

# eval_results = trainer.evaluate()
# print("\n✅ Final Evaluation Results:")
# print(f"Precision: {eval_results['eval_precision']:.4f}")
# print(f"Recall: {eval_results['eval_recall']:.4f}")
# print(f"F1 Score: {eval_results['eval_f1']:.4f}")
# print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")

# # ----------------------------------------
# # 🆕 NEW Inference Code
# # ----------------------------------------

# # Pre-load model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained("./ner_tokenizer")
# model = AutoModelForTokenClassification.from_pretrained("./ner_model")

# # Better Pipeline
# ner_pipeline = pipeline(
#     "ner",
#     model=model,
#     tokenizer=tokenizer,
#     aggregation_strategy="max"  # Better aggregation
# )

# def clean_text(text):
#     text = text.replace('\n', ' ')
#     text = text.replace('\t', ' ')
#     text = " ".join(text.split())
#     return text

# def extract_job_title_and_skills(text: str):
#     text = clean_text(text)
#     ner_results = ner_pipeline(text)

#     job_title = None
#     skills = []

#     for entity in ner_results:
#         label = entity["entity_group"]
#         word = entity["word"]

#         if label == "JOB_TITLE" and not job_title:
#             job_title = word
#         elif label == "SKILL":
#             skills.append(word)

#     if not job_title:
#         job_title = "Not detected"

#     return job_title, list(set(skills))  # Remove duplicate skills




import os
import sys
import torch
import pandas as pd
import numpy as np
from torch import nn
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    AutoConfig,
    pipeline,
)
from seqeval.metrics import classification_report
import ast
import re

# ----------------- Setup -----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))

# Corrected DATA_PATH to point to Backend\data
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # Points to Backend
DATA_PATH = os.path.join(PROJECT_ROOT, "data/IT jobs for training.csv")
SPLIT_DIR = os.path.join(PROJECT_ROOT, "data/split")
MODEL_DIR = os.path.join(BASE_DIR, "ner_model")
TOKENIZER_DIR = os.path.join(BASE_DIR, "ner_tokenizer")
os.makedirs(SPLIT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TOKENIZER_DIR, exist_ok=True)

# Verify dataset exists
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

# ----------------- Annotation -----------------
def annotate_skills_and_job_title(text, skills, job_title):
    entities = []
    if pd.isna(skills) or pd.isna(text) or pd.isna(job_title):
        return {'text': text, 'entities': []}

    text_lower = text.lower()
    
    # Annotate job title
    job_title_lower = job_title.lower().strip()
    start = text_lower.find(job_title_lower)
    if start != -1:
        end = start + len(job_title_lower)
        entities.append({'start': start, 'end': end, 'label': 'JOB_TITLE'})
    
    # Annotate skills
    skill_list = [s.strip() for s in skills.split(',')]
    for skill in skill_list:
        skill_lower = skill.lower()
        start = 0
        while True:
            start = text_lower.find(skill_lower, start)
            if start == -1:
                break
            end = start + len(skill)
            entities.append({'start': start, 'end': end, 'label': 'SKILL'})
            start = end

    # Sort entities by start position to avoid overlap issues
    entities.sort(key=lambda x: (x['start'], -x['end']))
    return {'text': text, 'entities': entities}

# ----------------- Load or Create Splits -----------------
if os.path.exists(f"{SPLIT_DIR}/train.csv") and os.path.exists(f"{SPLIT_DIR}/test.csv"):
    print(" Using previously saved train/test splits...")
    train_df = pd.read_csv(f"{SPLIT_DIR}/train.csv")
    test_df = pd.read_csv(f"{SPLIT_DIR}/test.csv")

    train_df['entities'] = train_df['entities'].apply(ast.literal_eval)
    test_df['entities'] = test_df['entities'].apply(ast.literal_eval)
else:
    print("🔄 Splitting and saving dataset for the first time...")
    df = pd.read_csv(DATA_PATH)
    annotated_data = [
        annotate_skills_and_job_title(row['job_description'], row['skills'], row['job_title'])
        for _, row in df.iterrows()
    ]
    full_df = pd.DataFrame(annotated_data)
    split = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_df = split.iloc[:int(0.8 * len(split))]
    test_df = split.iloc[int(0.8 * len(split)):]
    train_df.to_csv(f"{SPLIT_DIR}/train.csv", index=False)
    test_df.to_csv(f"{SPLIT_DIR}/test.csv", index=False)

# Convert to Hugging Face datasets
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "test": Dataset.from_pandas(test_df)
})

# ----------------- Tokenization -----------------
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
label_names = ["O", "B-SKILL", "I-SKILL", "B-JOB_TITLE", "I-JOB_TITLE"]
label2id = {label: idx for idx, label in enumerate(label_names)}
id2label = {idx: label for label, idx in label2id.items()}

def tokenize_and_align_labels(examples):
    tokenized = tokenizer(examples["text"], truncation=True, padding=True)
    labels = []

    for i, entities in enumerate(examples["entities"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = [-100] * len(word_ids)
        for entity in entities:
            for idx, word_id in enumerate(word_ids):
                if word_id is None:
                    continue
                char_span = tokenized.token_to_chars(i, idx)
                if char_span and entity["start"] <= char_span.start < entity["end"]:
                    label = entity["label"]
                    is_begin = char_span.start == entity["start"]
                    if label == "SKILL":
                        label_ids[idx] = label2id["B-SKILL"] if is_begin else label2id["I-SKILL"]
                    elif label == "JOB_TITLE":
                        label_ids[idx] = label2id["B-JOB_TITLE"] if is_begin else label2id["I-JOB_TITLE"]
        labels.append(label_ids)

    tokenized["labels"] = labels
    return tokenized

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# ----------------- Model -----------------
config = AutoConfig.from_pretrained(
    model_name,
    num_labels=len(label_names),
    id2label=id2label,
    label2id=label2id,
    hidden_dropout_prob=0.3,
    attention_probs_dropout_prob=0.3
)

model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs["logits"]
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.0, 5.0, 10.0, 5.0]).to(model.device))
        loss = loss_fct(logits.view(-1, len(label_names)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_preds = [[label_names[p] for (p, l) in zip(pred, label) if l != -100] for pred, label in zip(predictions, labels)]
    report = classification_report(true_labels, true_preds, output_dict=True)
    return {
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "f1": report["macro avg"]["f1-score"],
        "accuracy": np.mean([float(t == p) for t, p in zip(true_labels, true_preds)]),
    }

# ----------------- Training -----------------
training_args = TrainingArguments(
    output_dir="./ner_results",
    eval_strategy="epoch",
    learning_rate=2e-6,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir='./logs',
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()

model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(TOKENIZER_DIR)

eval_results = trainer.evaluate()
print("\n Final Evaluation Results:")
print(f"Precision: {eval_results['eval_precision']:.4f}")
print(f"Recall: {eval_results['eval_recall']:.4f}")
print(f"F1 Score: {eval_results['eval_f1']:.4f}")
print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")

# ----------------- Inference -----------------
# Pre-load model and tokenizer for testing
try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR, local_files_only=True)
except Exception as e:
    print(f"Error loading model/tokenizer: {str(e)}")
    raise

# Better Pipeline
ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)

def clean_text(text):
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'[^\w\s,]', '', text)
    text = " ".join(text.split())
    return text

def extract_job_title_and_skills(text: str):
    text = clean_text(text)
    ner_results = ner_pipeline(text)
    
    job_title = None
    skills = []
    
    for entity in ner_results:
        word = entity["word"].strip()
        if word.startswith("##") or len(word) < 2 or word.isdigit():
            continue
        if entity["entity_group"] == "JOB_TITLE" and not job_title:
            job_title = word
        elif entity["entity_group"] == "SKILL":
            skills.append(word)
    
    return job_title if job_title else "Not detected", list(set(skills))


