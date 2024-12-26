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
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModel, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import GroupKFold
import random
import os
from cuml.neighbors import NearestNeighbors
from transformers import set_seed, AutoConfig, AutoModel, MistralPreTrainedModel, MistralConfig, DynamicCache, \
    Cache
from typing import List, Tuple, Optional, Union
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa, \
    _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers import BitsAndBytesConfig
# from transformers.models.mistral.modeling_flax_mistral import MISTRAL_INPUTS_DOCSTRING
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer, MistralRMSNorm
from transformers.utils import add_start_docstrings_to_model_forward
from torch import nn, Tensor
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

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
exp = "239"
exp_dir = OUTPUT_DIR / "exp" / f"ex{exp}"
model_dir = exp_dir / "model"

exp_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)
logger_path = exp_dir / f"ex{exp}.txt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# mdoel settings
# =========================

seed = 1
model_path = "Salesforce/SFR-Embedding-2_R"
batch_size = 30
n_epochs = 10
max_len = 144
weight_decay = 0.1
lr = 1e-4
num_warmup_steps_rate = 0.1
tokenizer = AutoTokenizer.from_pretrained(model_path)
n_candidate = 25
iters_to_accumulate = 1


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


class MistralModel(MistralPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MistralDecoderLayer`]

    Args:
        config: MistralConfig
    """

    def __init__(self, config: MistralConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MistralDecoderLayer(config, layer_idx)
             for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # @add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        labels: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                # logger.warning_once(
                #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                # )
                use_cache = False

        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(
                    past_key_values)
            past_key_values_length = past_key_values.get_usable_length(
                seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Mistral. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (
                attention_mask is not None and 0 in attention_mask) else None
        elif self._attn_implementation == "sdpa" and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache(
            ) if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class BiEncoderModel(nn.Module):
    def __init__(self,
                 sentence_pooling_method: str = "last"
                 ):
        super().__init__()
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )
        model = MistralModel.from_pretrained(
            model_path, quantization_config=bnb_config)
        # model = IgnoreLabelsWrapper(model)
        config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
            lora_dropout=0.05,  # Conventional
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(model, config)
        # self.model.gradient_checkpointing_enable()
        self.model.print_trainable_parameters()
        self.sentence_pooling_method = sentence_pooling_method
        self.config = self.model.config

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def last_token_pool(self, last_hidden_states: Tensor,
                        attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def sentence_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == 'cls':
            return hidden_state[:, 0]
        elif self.sentence_pooling_method == 'last':
            return self.last_token_pool(hidden_state, mask)

    def encode(self, input_is, attention_mask):
        # print(features)
        psg_out = self.model(input_ids=input_is, attention_mask=attention_mask,
                             return_dict=True)
        p_reps = self.sentence_embedding(
            psg_out.last_hidden_state, attention_mask)
        return p_reps.contiguous()

    def forward(self, input_is, attention_mask):
        q_reps = self.encode(input_is, attention_mask)
        return q_reps

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors


def get_optimizer_grouped_parameters(
        model,
        weight_decay,
        lora_lr=5e-4,
        no_decay_name_list=["bias", "LayerNorm.weight"],
        lora_name_list=["lora_right_weight", "lora_left_weight"],
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list)
                    and p.requires_grad and not any(nd in n
                                                    for nd in lora_name_list))
            ],
            "weight_decay":
                weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list)
                    and p.requires_grad and any(nd in n
                                                for nd in lora_name_list))
            ],
            "weight_decay":
                weight_decay,
            "lr":
                lora_lr
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
                0.0,
        },
    ]
    if not optimizer_grouped_parameters[1]["params"]:
        optimizer_grouped_parameters.pop(1)
    return optimizer_grouped_parameters


def save_model(output_dir, model, tokenizer, fold):
    save_dir = output_dir / f"fold{fold}"
    save_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(save_dir)
    # model = convert_lora_to_linear_layer(model)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save = model_to_save.model
    # model_to_save.save_pretrained(output_dir)

    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "adapter.bin"
    output_model_file = save_dir / WEIGHTS_NAME
    save_dict = model_to_save.state_dict()
    final_d = {}
    for k, v in save_dict.items():
        if "lora" in k:
            final_d[k] = v
    torch.save(final_d, output_model_file)
    print('saving success')


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


task = 'Given a math problem statement and an incorrect answer as a query, retrieve relevant passages that identify and explain the nature of the error.'


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
    llm_text[["QuestionId", "ans", "llmMisconception"]], how="left", on=["QuestionId", "ans"])

train_pivot["all_text"] = ' <Question> ' + train_pivot['QuestionText'] + \
    ' <Answer> ' + train_pivot['AnswerText'] + \
    '<Construct> ' + train_pivot['ConstructName'] + \
                          ' <Subject> ' + train_pivot['SubjectName'] + \
                          ' <LLMOutput> ' + train_pivot['llmMisconception']
train_pivot["MisconceptionId"] = train_pivot["MisconceptionId"].astype(int)
train_pivot = train_pivot.merge(
    misconception, how="left", on="MisconceptionId")

text_list = []
for t in train_pivot["all_text"].values:
    text_list.append(get_detailed_instruct(task, t))
train_pivot["all_text"] = text_list
df_fold = df_fold.drop_duplicates(subset=["QuestionId"]).reset_index(drop=True)
train_pivot = train_pivot.merge(
    df_fold[["QuestionId", "fold"]], how="left", on="QuestionId")
fold_array = train_pivot["fold"].values

# ================================
# train
# ================================
with timer("train"):
    set_seed(seed)
    gkf = GroupKFold(n_splits=5)
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
                                  num_workers=0)

        model = BiEncoderModel()
        # model.gradient_checkpointing_enable()
        model = model.to(device)

        # optimizer, scheduler
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(
            model, 0.01, 5e-4)

        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=lr,
                          betas=(0.9, 0.95),
                          fused=True)
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

                ids2 = d2["input_ids"].to(device)
                mask2 = d2['attention_mask'].to(device)
                with autocast():
                    output1 = model(ids1, mask1)
                    output2 = model(ids2, mask2)
                    loss = criterion(output1,
                                     output2)
                    loss = loss / iters_to_accumulate
                    train_losses_batch.append(loss.item())
                scaler.scale(loss).backward()
                if (i + 1) % iters_to_accumulate == 0:
                    # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

            train_loss_mean = np.mean(train_losses_batch)
            LOGGER.info(f'fold {n} epoch :{epoch} loss {train_loss_mean}')

            # val
            model.eval()
            pred = make_candidate_first_stage_for_val(x_val, misconception,
                                                      model, tokenizer, max_len*2,
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
                save_model(model_dir, model, tokenizer, n)
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
