# Import comet_ml in the top of your file
from comet_ml import Experiment
# Create an experiment
experiment = Experiment(api_key="9gCeZhtoyUQ3YKvSQlzc6wjcH",
                        project_name="general", workspace="shizhediao")

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import tokenizers
import transformers
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm.autonotebook import tqdm

import utils
from dataset import TweetDataset
from model import TweetModel
import argparse

from torch.utils.data import RandomSampler, SequentialSampler

parser = argparse.ArgumentParser(description='robert4qa')
parser.add_argument('--max_len', type=int, default=160,
                    help='maximum length')
parser.add_argument('--train_batch_size', type=int, default=96,
                    help='maximum length')
parser.add_argument('--valid_batch_size', type=int, default=32,
                    help='maximum length')
parser.add_argument('--epochs', type=int, default=10,
                    help='maximum length')
parser.add_argument('--lr', type=float, default=3e-5,
                    help='lr')
parser.add_argument('--patience', type=int, default=3,
                    help='patience')
parser.add_argument('--num_warmup_steps', type=int, default=200,
                    help='num_warmup_steps')
parser.add_argument('--shuffle', action='store_true',
                    help='shuffle or not')


args = parser.parse_args()
args.parallel = True
print(vars(args))
experiment.log_parameters(vars(args))

max_len = args.max_len
train_batch_size = args.train_batch_size
valid_batch_size = args.valid_batch_size
epochs = args.epochs


def cal_loss(start_logits, end_logits, start_positions, end_positions):
    loss_fn = nn.CrossEntropyLoss()
    start_loss = loss_fn(start_logits, start_positions)    # 情感文本首词概率
    end_loss = loss_fn(end_logits, end_positions)    # 情感文本末词概率
    total_loss = start_loss + end_loss    # 总损失
    return total_loss

def cal_jaccard(tweet, target, idx_start, idx_end, input_offsets):
    if idx_end < idx_start:
        idx_end = idx_start
    output = ""
    for idx in range(idx_start, idx_end + 1):
        output += tweet[input_offsets[idx][0]: input_offsets[idx][1]]
    jac = utils.jaccard(target, output)
    return jac, output

def train_fn(data_loader, model, optimizer, device, scheduler = None):
    model.train()    # 训练模式

    losses = utils.AverageMeter()
    jaccards = utils.AverageMeter()

    tk = tqdm(data_loader, total = len(data_loader))
    # 按batch读取
    for bi, d in enumerate(tk):
        input_ids = d["input_ids"].to(device, dtype = torch.long)
        mask = d["mask"].to(device, dtype = torch.long)
        token_type = d["token_type"].to(device, dtype = torch.long)
        idx_word_start = d["idx_word_start"].to(device, dtype = torch.long)
        idx_word_end = d["idx_word_end"].to(device, dtype = torch.long)
        tweet =  d["tweet"]
        selected_text = d["selected_text"]
        input_offsets = d["input_offsets"]

        model.zero_grad()
        # batch_size x 序列长度(160)，batch_size x 序列长度(192)
        outputs_start, outputs_end = model(input_ids = input_ids, mask = mask, token_type = token_type)
        loss = cal_loss(outputs_start, outputs_end, idx_word_start, idx_word_end)
        if args.parallel:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        loss.backward()
        optimizer.step()
        scheduler.step()

        outputs_start = torch.softmax(outputs_start, dim = 1).cpu().detach().numpy()
        outputs_end = torch.softmax(outputs_end, dim = 1).cpu().detach().numpy()

        jaccard_scores = []
        for i, twee in enumerate(tweet):
            selected_tweet = selected_text[i]
            jaccard_score, _ = cal_jaccard(twee, selected_tweet, np.argmax(outputs_start[i, :]), np.argmax(outputs_end[i, :]), input_offsets[i])
            jaccard_scores.append(jaccard_score)
        # 打印loss和jaccard
        jaccards.update(np.mean(jaccard_scores), input_ids.size(0))
        losses.update(loss.item(), input_ids.size(0))
        tk.set_postfix(loss = losses.avg, jaccard = jaccards.avg)

def eval_fn(data_loader, model, device):
    model.eval()    # 测试模式

    losses = utils.AverageMeter()
    jaccards = utils.AverageMeter()

    with torch.no_grad():
        tk = tqdm(data_loader, total = len(data_loader))
        # 按batch读取
        for bi, d in enumerate(tk):
            input_ids = d["input_ids"].to(device, dtype=torch.long)
            mask = d["mask"].to(device, dtype=torch.long)
            token_type = d["token_type"].to(device, dtype=torch.long)
            idx_word_start = d["idx_word_start"].to(device, dtype=torch.long)
            idx_word_end = d["idx_word_end"].to(device, dtype=torch.long)
            tweet = d["tweet"]
            selected_text = d["selected_text"]
            input_offsets = d["input_offsets"]

            # batch_size x 序列长度(192)，batch_size x 序列长度(192)
            outputs_start, outputs_end = model(input_ids=input_ids, mask=mask, token_type=token_type)
            loss = cal_loss(outputs_start, outputs_end, idx_word_start, idx_word_end)

            outputs_start = torch.softmax(outputs_start, dim = 1).cpu().detach().numpy()
            outputs_end = torch.softmax(outputs_end, dim = 1).cpu().detach().numpy()

            jaccard_scores = []
            for i, twee in enumerate(tweet):
                selected_tweet = selected_text[i]
                jaccard_score, _ = cal_jaccard(twee, selected_tweet, np.argmax(outputs_start[i, :]),
                                               np.argmax(outputs_end[i, :]), input_offsets[i])
                jaccard_scores.append(jaccard_score)
            # 打印loss和jaccard
            jaccards.update(np.mean(jaccard_scores), input_ids.size(0))
            losses.update(loss.item(), input_ids.size(0))
            tk.set_postfix(loss=losses.avg, jaccard=jaccards.avg)
    print("Jaccard = ", jaccards.avg)
    return jaccards.avg

def train(fold, epochs, training_file, tokenizer, max_len, train_batch_size, valid_batch_size, roberta_path, lr, patience, num_warmup_steps):
    dfx = pd.read_csv(training_file)

    df_train = dfx[dfx.kfold != fold].reset_index(drop = True)
    df_valid = dfx[dfx.kfold == fold].reset_index(drop = True)

    # 训练集
    train_dataset = TweetDataset(
        tweet = df_train.text.values,
        sentiment = df_train.sentiment.values,
        selected_text = df_train.selected_text.values,
        tokenizer = tokenizer,
        max_len = max_len
    )
    # 验证集
    valid_dataset = TweetDataset(
        tweet = df_valid.text.values,
        sentiment = df_valid.sentiment.values,
        selected_text = df_valid.selected_text.values,
        tokenizer = tokenizer,
        max_len = max_len
    )

    train_sampler, valid_sampler = None, None
    if args.shuffle:
        train_sampler = RandomSampler(train_dataset)
        valid_sampler = SequentialSampler(valid_dataset)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = train_batch_size,
        num_workers = 4,
        sampler=train_sampler
    )


    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size = valid_batch_size,
        num_workers = 2,
        sampler=valid_sampler
    )

    device = torch.device("cuda")

    model_config = transformers.RobertaConfig.from_pretrained(roberta_path)
    model_config.output_hidden_states = True
    model = TweetModel(roberta_path = roberta_path, conf = model_config)
    model.to(device)

    num_train_steps = int(len(df_train) / train_batch_size * epochs)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    optimizer = AdamW(optimizer_parameters, lr = lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = num_warmup_steps,
        num_training_steps = num_train_steps
    )

    # multi-gpu training (should be after apex fp16 initialization)
    if args.parallel:
        model = torch.nn.DataParallel(model)

    es = utils.EarlyStopping(patience = patience, mode = "max")
    print("Training is Starting for fold", fold)

    for epoch in range(epochs):
        train_fn(train_data_loader, model, optimizer, device, scheduler = scheduler)
        jaccard = eval_fn(valid_data_loader, model, device)
        print("Jaccard Score = ", jaccard)
        experiment.log_metric("jaccard", jaccard)
        es(jaccard, model, model_path = f"./bin/model_{fold}.bin")
        if es.early_stop:
            print("Early stopping")
            break
    del model, optimizer, scheduler, df_train, df_valid, train_dataset, valid_dataset, train_data_loader, valid_data_loader
    import gc
    gc.collect()
    torch.cuda.empty_cache()

# max_len = 160
# train_batch_size = 16
# valid_batch_size = 8
# epochs = 3

roberta_path = "./roberta-base"
training_file = "./train-kfolds/train_5folds.csv"

tokenizer = tokenizers.ByteLevelBPETokenizer(
    vocab_file = "./roberta-base/vocab.json",
    merges_file = "./roberta-base/merges.txt",
    lowercase = True,
    add_prefix_space = True
)

# train(0, epochs, training_file, tokenizer, max_len, train_batch_size, valid_batch_size, roberta_path)
# train(1, epochs, training_file, tokenizer, max_len, train_batch_size, valid_batch_size, roberta_path)
# train(2, epochs, training_file, tokenizer, max_len, train_batch_size, valid_batch_size, roberta_path)
# train(3, epochs, training_file, tokenizer, max_len, train_batch_size, valid_batch_size, roberta_path)
# train(4, epochs, training_file, tokenizer, max_len, train_batch_size, valid_batch_size, roberta_path)
train(0, epochs, training_file, tokenizer, max_len, train_batch_size, valid_batch_size, roberta_path, args.lr, args.patience, args.num_warmup_steps)
train(1, epochs, training_file, tokenizer, max_len, train_batch_size, valid_batch_size, roberta_path, args.lr, args.patience, args.num_warmup_steps)
train(2, epochs, training_file, tokenizer, max_len, train_batch_size, valid_batch_size, roberta_path, args.lr, args.patience, args.num_warmup_steps)
train(3, epochs, training_file, tokenizer, max_len, train_batch_size, valid_batch_size, roberta_path, args.lr, args.patience, args.num_warmup_steps)
train(4, epochs, training_file, tokenizer, max_len, train_batch_size, valid_batch_size, roberta_path, args.lr, args.patience, args.num_warmup_steps)
