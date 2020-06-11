import torch
import torch.nn as nn
import transformers


class TweetModel(transformers.BertPreTrainedModel):
    def __init__(self, roberta_path, conf):
        super(TweetModel, self).__init__(conf)

        # 预训练的roberta模型
        self.roberta = transformers.RobertaModel.from_pretrained(roberta_path, config = conf)
        self.drop_out = nn.Dropout(0.5)

        # bert多层融合
        self.w0 = nn.Linear(768 * 6, 768 * 6)

        # roberta-base隐藏状态的维度是768
        self.lstm = nn.LSTM(input_size = 768 * 6, hidden_size = 768, num_layers = 1, bidirectional = True, batch_first = True)

        # 两维（情感文本首词概率，情感文本末词概率）
        self.l0 = nn.Linear(768 * 2, 2)

    def forward(self, input_ids, mask, token_type):
        self.lstm.flatten_parameters()
        # bert层数 x batch_size x 序列长度(160) x 768
        _, _, out = self.roberta(input_ids, attention_mask = mask, token_type_ids = token_type)

        # batch_size x 序列长度(160) x (768*6)
        out = torch.cat((out[-1], out[-2], out[-3], out[4], out[5], out[0]), dim = -1)
        out = self.drop_out(out)
        out = self.w0(out)

        out, _ = self.lstm(out)

        # batch_size x 序列长度(160) x 2
        logits = self.l0(out)

        # batch_size x 序列长度(160) x 2 -> batch_size x 序列长度(160) x 1，batch_size x 序列长度(160) x 1
        start_logits, end_logits = logits.split(1, dim = -1)

        # batch_size x 序列长度(160)
        start_logits = start_logits.squeeze(-1)

        # batch_size x 序列长度(160)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits