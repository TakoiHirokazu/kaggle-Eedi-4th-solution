# =========================
# libraries
# =========================
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import time
import logging
from contextlib import contextmanager
import sys
from transformers import AutoModel, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import GroupKFold
import random
import os
from cuml.neighbors import NearestNeighbors
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# =========================
# constants
# =========================
DATA_DIR = Path("/tmp/working/data")
OUTPUT_DIR = Path("/tmp/working/storage/eedi/output/")
TRAIN_PATH = DATA_DIR / "train.csv"
MISCONCEPTION_MAPPING_PATH = DATA_DIR / "misconception_mapping.csv"
LLM_TEXT_PATH = Path(
    "/tmp/working/output/kaggle/exp105/exp105_train_add_text.csv")
FOLD_PATH = "/tmp/working/output/team/eedi_fold.csv"

# =========================
# settings
# =========================
exp = "241"
exp_dir = OUTPUT_DIR / "exp" / f"ex{exp}"
model_dir = exp_dir / "model"

exp_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)
logger_path = exp_dir / f"ex{exp}.txt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# mdoel settings
# =========================
seed = 0
model_path = "Alibaba-NLP/gte-large-en-v1.5"
batch_size = 60
iters_to_accumulate = 1
n_epochs = 10
max_len = 400
weight_decay = 0.1
beta = (0.9, 0.98)
lr = 5e-5
num_warmup_steps_rate = 0.1
tokenizer = AutoTokenizer.from_pretrained(model_path)
n_candidate = 25


# ===============
# Functions
# ===============
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EediDataset(Dataset):
    def __init__(self, text1, text2,
                 tokenizer, max_len,
                 labels=None):
        self.text1 = text1
        self.text2 = text2
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels = labels

    def __len__(self):
        return len(self.text1)

    def __getitem__(self, item):
        text1 = self.text1[item]
        text2 = self.text2[item]
        inputs1 = self.tokenizer(
            text1,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True
        )

        inputs2 = self.tokenizer(
            text2,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True
        )

        inputs1 = {"input_ids": torch.tensor(inputs1["input_ids"],
                                             dtype=torch.long),
                   "attention_mask": torch.tensor(inputs1["attention_mask"],
                                                  dtype=torch.long),
                   "token_type_ids": torch.tensor(inputs1["token_type_ids"],
                                                  dtype=torch.long)}
        inputs2 = {"input_ids": torch.tensor(inputs2["input_ids"], dtype=torch.long),
                   "attention_mask": torch.tensor(inputs2["attention_mask"],
                                                  dtype=torch.long),
                   "token_type_ids": torch.tensor(inputs2["token_type_ids"],
                                                  dtype=torch.long)}

        if self.labels is not None:
            label = self.labels[item]
            return {
                "input1": inputs1,
                "input2": inputs2,
                "label": torch.tensor(label, dtype=torch.float32),
            }
        else:
            return {
                "input1": inputs1,
                "input2": inputs2,
            }


class EediValDataset(Dataset):
    def __init__(self, text1,
                 tokenizer, max_len):
        self.text1 = text1
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text1)

    def __getitem__(self, item):
        text1 = self.text1[item]
        inputs1 = self.tokenizer(
            text1,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True
        )
        inputs1 = {"input_ids": torch.tensor(inputs1["input_ids"], dtype=torch.long),
                   "attention_mask": torch.tensor(inputs1["attention_mask"],
                                                  dtype=torch.long),
                   "token_type_ids": torch.tensor(inputs1["token_type_ids"],
                                                  dtype=torch.long)}

        return inputs1


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class SentenceBertModel(nn.Module):
    def __init__(self):
        super(SentenceBertModel, self).__init__()
        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True)
        self.pool = MeanPooling()

    def forward(self, ids, mask):
        # pooler
        out = self.model(ids,
                         attention_mask=mask)['last_hidden_state']
        out = self.pool(out, mask)
        return out


def cos_sim(a, b):
    # From https://github.com/UKPLab/sentence-transformers/blob/master/
    # sentence_transformers/util.py#L31
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


class MultipleNegativesRankingLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, embeddings_a, embeddings_b, labels=None):
        """
        Compute similarity between `a` and `b`.
        Labels have the index of the row number at each row.
        This indicates that `a_i` and `b_j` have high similarity
        when `i==j` and low similarity when `i!=j`.
        """

        similarity_scores = (
            cos_sim(embeddings_a, embeddings_b) * 20.0
        )  # Not too sure why to scale it by 20:
        # https://github.com/UKPLab/sentence-transformers/
        # blob/b86eec31cf0a102ad786ba1ff31bfeb4998d3ca5/sentence_transformers/
        # losses/MultipleNegativesRankingLoss.py#L57

        labels = torch.tensor(
            range(len(similarity_scores)),
            dtype=torch.long,
            device=similarity_scores.device,
        )  # Example a[i] should match with b[i]

        return self.loss_function(similarity_scores, labels)


def collate_sentence(d):
    mask_len = int(d["attention_mask"].sum(axis=1).max())
    return {"input_ids": d['input_ids'][:, :mask_len],
            "attention_mask": d['attention_mask'][:, :mask_len],
            "token_type_ids": d["token_type_ids"][:, :mask_len]}


def make_candidate_first_stage_for_val(val, misconception,
                                       model, tokenizer, max_len,
                                       batch_size, n_neighbor=100):
    val_ = EediValDataset(val["all_text"],
                          tokenizer,
                          max_len)
    misconception_ = EediValDataset(misconception["MisconceptionName"],
                                    tokenizer,
                                    max_len)
    # make val emb
    val_loader = DataLoader(
        val_, batch_size=batch_size * 2, shuffle=False)
    val_emb = make_emb(model, val_loader)

    # make misconception emb
    misconcept_loader = DataLoader(
        misconception_, batch_size=batch_size * 2, shuffle=False)
    misconcept_emb = make_emb(model, misconcept_loader)

    knn = NearestNeighbors(n_neighbors=n_neighbor,
                           metric="cosine")
    knn.fit(misconcept_emb)
    dists, nears = knn.kneighbors(val_emb)
    return nears


def make_emb(model, train_loader):
    bert_emb = []
    with torch.no_grad():
        for d in train_loader:
            d = collate_sentence(d)
            input_ids = d['input_ids']
            mask = d['attention_mask']
            input_ids = input_ids.to(device)
            mask = mask.to(device)
            output = model(input_ids, mask)
            output = output.detach().cpu().numpy().astype(np.float32)
            bert_emb.append(output)
    torch.cuda.empty_cache()
    bert_emb = np.concatenate(bert_emb)
    return bert_emb


def calculate_map25_with_metrics(df):
    def ap_at_k(actual, predicted, k=25):
        actual = int(actual)
        predicted = predicted[:k]
        score = 0.0
        num_hits = 0.0
        found = False
        rank = None
        for i, p in enumerate(predicted):
            if p == actual:
                if not found:
                    found = True
                    rank = i + 1
                num_hits += 1
                score += num_hits / (i + 1.0)
        return score, found, rank

    scores = []
    found_count = 0
    rankings = []
    total_count = 0

    for _, row in df.iterrows():
        actual = row['MisconceptionId']
        predicted = [int(float(x)) for x in row['pred'].split()]
        score, found, rank = ap_at_k(actual, predicted)
        scores.append(score)

        total_count += 1
        if found:
            found_count += 1
            rankings.append(rank)

    map25 = np.mean(scores)
    percent_found = (found_count / total_count) * 100 if total_count > 0 else 0
    avg_ranking = np.mean(rankings) if rankings else 0

    return map25, percent_found, avg_ranking


LOGGER = logging.getLogger()
FORMATTER = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")


def setup_logger(out_file=None, stderr=True,
                 stderr_level=logging.INFO, file_level=logging.DEBUG):
    LOGGER.handlers = []
    LOGGER.setLevel(min(stderr_level, file_level))

    if stderr:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(FORMATTER)
        handler.setLevel(stderr_level)
        LOGGER.addHandler(handler)

    if out_file is not None:
        handler = logging.FileHandler(out_file)
        handler.setFormatter(FORMATTER)
        handler.setLevel(file_level)
        LOGGER.addHandler(handler)

    LOGGER.info("logger set up")
    return LOGGER


@ contextmanager
def timer(name):
    t0 = time.time()
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s')


setup_logger(out_file=logger_path)


# ============================
# main
# ============================
train = pd.read_csv(TRAIN_PATH)
misconception = pd.read_csv(MISCONCEPTION_MAPPING_PATH)
llm_text = pd.read_csv(LLM_TEXT_PATH)
df_fold = pd.read_csv(FOLD_PATH)
train_pivot = []
common_cols = ['QuestionId', 'ConstructId', 'ConstructName', 'SubjectId',
               'SubjectName', 'CorrectAnswer', 'QuestionText']
for i in ["A", "B", "C", "D"]:
    train_ = train.copy()
    train_ = train[common_cols + [f"Answer{i}Text", f"Misconception{i}Id"]]
    train_ = train_.rename({f"Answer{i}Text": "AnswerText",
                            f"Misconception{i}Id": "MisconceptionId"}, axis=1)
    train_["ans"] = i
    train_pivot.append(train_)

train_pivot = pd.concat(train_pivot).reset_index(drop=True)
train_pivot = train_pivot[train_pivot["MisconceptionId"].notnull()].reset_index(
    drop=True)
train_pivot = train_pivot.merge(
    llm_text[["QuestionId", "ans", "llmMisconception"]], how="left", on=[
        "QuestionId", "ans"])
train_pivot["all_text"] = '<Construct> ' + train_pivot['ConstructName'] + \
                          ' <Subject> ' + train_pivot['SubjectName'] + \
    ' <Question> ' + train_pivot['QuestionText'] + \
    ' <Answer> ' + train_pivot['AnswerText'] + \
    ' <LLM OUTPUT> ' + train_pivot['llmMisconception']

train_pivot["MisconceptionId"] = train_pivot["MisconceptionId"].astype(int)
train_pivot = train_pivot.merge(
    misconception, how="left", on="MisconceptionId")
df_fold = df_fold.drop_duplicates(subset=["QuestionId"]).reset_index(drop=True)
train_pivot = train_pivot.merge(
    df_fold[["QuestionId", "fold"]], how="left", on="QuestionId")
fold_array = train_pivot["fold"].values
# ================================
# train
# ================================
with timer("train"):
    set_seed(seed)
    val_pred_all = []
    for n in range(5):
        x_train = train_pivot[fold_array != n].reset_index(drop=True)
        x_val = train_pivot[fold_array == n].reset_index(drop=True)
        train_ = EediDataset(x_train["all_text"],
                             x_train["MisconceptionName"],
                             tokenizer,
                             max_len,
                             x_train["MisconceptionId"])

        # loader
        train_loader = DataLoader(dataset=train_,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=4)

        model = SentenceBertModel()
        model = model.to(device)

        # optimizer, scheduler
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=lr,
                          betas=beta,
                          weight_decay=weight_decay,
                          )
        num_train_optimization_steps = int(len(train_loader) * n_epochs)
        num_warmup_steps = int(
            num_train_optimization_steps * num_warmup_steps_rate)
        scheduler = \
            get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_train_optimization_steps)

        criterion = MultipleNegativesRankingLoss()
        best_score = 0
        scaler = GradScaler()
        for epoch in range(n_epochs):
            print(f"============start epoch:{epoch}==============")
            model.train()
            train_losses_batch = []
            for i, d in tqdm(enumerate(train_loader), total=len(train_loader)):
                d1 = d["input1"]
                d2 = d["input2"]
                d1 = collate_sentence(d1)
                d2 = collate_sentence(d2)

                ids1 = d1["input_ids"].to(device)
                mask1 = d1['attention_mask'].to(device)
                token_type_ids1 = d1["token_type_ids"].to(device)

                ids2 = d2["input_ids"].to(device)
                mask2 = d2['attention_mask'].to(device)
                token_type_ids2 = d2["token_type_ids"].to(device)
                with autocast():
                    output1 = model(ids1, mask1)
                    output2 = model(ids2, mask2)
                    loss = criterion(output1,
                                     output2)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                train_losses_batch.append(loss.item())
            train_loss_mean = np.mean(train_losses_batch)
            LOGGER.info(f'fold {n} epoch :{epoch} loss {train_loss_mean}')

            # val
            model.eval()
            pred = make_candidate_first_stage_for_val(x_val, misconception,
                                                      model, tokenizer, max_len,
                                                      batch_size, n_candidate)
            recall = 0
            for gt, p in zip(x_val["MisconceptionId"], pred):
                if gt in p:
                    recall += 1
            recall /= len(x_val)
            pred_ = []
            for i in pred:
                pred_.append(' '.join(map(str, i)))

            val_pred = pd.DataFrame()
            val_pred["MisconceptionId"] = x_val["MisconceptionId"]
            val_pred["pred"] = pred_
            val_pred["QuestionId"] = x_val["QuestionId"]
            val_pred["ans"] = x_val["ans"]
            val_score, percent_found, avg_ranking = calculate_map25_with_metrics(
                val_pred)
            LOGGER.info(
                f'fold {n} epoch :{epoch} cv {val_score} recall : {recall}')
            if recall > best_score:
                LOGGER.info(f'fold {n} epoch :{epoch} model save')
                torch.save(model.state_dict(), model_dir / f"exp{exp}_{n}.pth")
                best_val_pred = val_pred.copy()
                best_score = recall
        val_pred_all.append(best_val_pred)
val_pred_all = pd.concat(val_pred_all).reset_index(drop=True)
val_pred_all.to_parquet(exp_dir / f"exp{exp}_val_pred.parquet")
LOGGER.info(f'cv : {calculate_map25_with_metrics(val_pred_all)}')
kaggle_json = {
    "title": f"eedi-exp{exp}",
    "id": f"takoihiraokazu/eedi-exp{exp}",
    "licenses": [
        {
            "name": "CC0-1.0"
        }
    ]
}

with open(model_dir / "dataset-metadata.json",
          'w') as f:
    json.dump(kaggle_json, f)
