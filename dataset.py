import torch


def process_data(tweet, selected_text, sentiment, tokenizer, max_len):
    tweet = " " + " ".join(str(tweet).split())    # 推特文本的第一个词前加空格
    selected_text = " " + " ".join(str(selected_text).split())    # 情感文本的第一个词前加空格

    len_st = len(selected_text) - 1    # 情感文本长度（去除首空格）
    idx_char_start = -1    # 情感文本首词在推特文本中的索引
    idx_char_end = -1    # 情感文本末词在推特文本中的索引

    # 情感文本范围（字符级）
    for i in [index for index, c in enumerate(tweet) if c == selected_text[1]]:
        if " " + tweet[i: i + len_st] == selected_text:    # 左闭右开
            idx_char_start = i
            idx_char_end = i + len_st - 1
            break

    # 情感文本的字符，在推特文本中的索引对应的值被设为1（字符级）
    char_targets = [0] * len(tweet)
    if idx_char_start != -1 and idx_char_end != -1:
        for i in range(idx_char_start, idx_char_end + 1):    # 左闭右开
            char_targets[i]= 1

    tok_tweet = tokenizer.encode(tweet)    # 分词后的推特文本（词级）
    # print(tok_tweet.tokens)
    input_ids_orig = tok_tweet.ids    # 各词语id
    input_offsets_orig = tok_tweet.offsets    # 各词语首尾跨度（左闭右开）
    # print(tweet_offsets_orig)

    # 情感文本的词，在推特文本中的索引（词级）
    idx_word = []
    for i, (offset1, offset2) in enumerate(input_offsets_orig):
        # 情感文本的词
        if sum(char_targets[offset1: offset2]) > 0:
            idx_word.append(i)

    # 情感文本的首尾索引
    idx_word_start = idx_word[0]
    idx_word_end = idx_word[-1]

    # print("positive", tokenizer.encode("positive").ids)
    # print("negative", tokenizer.encode("negative").ids)
    # print("neutral", tokenizer.encode("neutral").ids)
    sentiment_id = {"positive": 1313, "negative": 2430, "neutral": 7974}

    input_ids = [0] + [sentiment_id[sentiment]] + [2] + [0] + input_ids_orig + [2]    # "<s>"：0，"</s>"：2

    idx_word_start += 4
    idx_word_end += 4

    token_type = [0] * 4 + [0] * (len(input_ids_orig) + 1)
    mask = [1] * len(token_type)
    input_offsets = [(0, 0)] * 4 + input_offsets_orig + [(0, 0)]

    padding_len = max_len - len(input_ids)
    if padding_len > 0:
        input_ids = input_ids + ([1] * padding_len)    # "<pad>"：1
        mask = mask + ([0] * padding_len)    # padding部分设置为0
        token_type = token_type + ([0] * padding_len)
        input_offsets = input_offsets + ([(0, 0)] * padding_len)

    return {
        "input_ids": input_ids,    # 各词语id
        "mask": mask,    # mask向量
        "token_type": token_type,
        "idx_word_start": idx_word_start,    # 情感文本首词索引
        "idx_word_end": idx_word_end,    # 情感文本末词索引
        "tweet": tweet,    # 推特文本
        "selected_text": selected_text,    # 情感文本
        "sentiment": sentiment,    # 情感标签
        "input_offsets": input_offsets    # 各词语首尾跨度
    }


class TweetDataset:
    def __init__(self, tweet, sentiment, selected_text, tokenizer, max_len):
        self.tweet = tweet    # 推特文本
        self.sentiment = sentiment    # 情感标签
        self.seleted_text = selected_text    # 情感文本
        self.tokenizer = tokenizer    # 分词器
        self.max_len = max_len    # 最大长度

    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):
        data = process_data(
            self.tweet[item],
            self.seleted_text[item],
            self.sentiment[item],
            self.tokenizer,
            self.max_len
        )
        return {
            "input_ids": torch.tensor(data["input_ids"], dtype = torch.long),
            "mask": torch.tensor(data["mask"], dtype = torch.long),
            "token_type": torch.tensor(data["token_type"], dtype = torch.long),
            "idx_word_start": torch.tensor(data["idx_word_start"], dtype = torch.long),
            "idx_word_end": torch.tensor(data["idx_word_end"], dtype = torch.long),
            "tweet": data["tweet"],
            "selected_text": data["selected_text"],
            "sentiment": data["sentiment"],
            "input_offsets": torch.tensor(data["input_offsets"], dtype = torch.long)
        }