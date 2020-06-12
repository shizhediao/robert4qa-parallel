import torch
import torch.nn as nn
import transformers

hidden_size = 768
class TweetModel(transformers.BertPreTrainedModel):
    def __init__(self, roberta_path, conf):
        super(TweetModel, self).__init__(conf)

        # 预训练的roberta模型
        self.roberta = transformers.RobertaModel.from_pretrained(roberta_path, config = conf)
        # self.roberta = AutoModelWithLMHead.from_pretrained("xlm-roberta-large")

        self.drop_out = nn.Dropout(0.5)

        # bert多层融合
        # self.w0 = nn.Linear(hidden_size * 12, hidden_size * 12)
        self.w0 = nn.Linear(hidden_size * 13, hidden_size  * 13)

        # roberta-base隐藏状态的维度是768
        self.lstm = nn.LSTM(input_size = hidden_size * 13, hidden_size = hidden_size, num_layers = 1, bidirectional = True, batch_first = True)

        # 两维（情感文本首词概率，情感文本末词概率）
        self.l0 = nn.Linear(hidden_size * 2, 2)
        torch.nn.init.normal_(self.l0.weight, std=0.02)

        self.ensemble_weight = nn.Linear(13, 1)


    def forward(self, input_ids, mask, token_type):
        self.lstm.flatten_parameters()
        # bert层数 x batch_size x 序列长度(160) x 768 = 13 x 24 x 160 x 768
        _, _, out = self.roberta(input_ids, attention_mask = mask, token_type_ids = token_type)

        # out = torch.cat((out[-1], out[-2], out[-3], out[4], out[5], out[0]), dim = -1)
        # out = nn.functional.softmax(self.ensemble_weight.weight, dim=1).mul(torch.stack(out))
        # weight = nn.functional.softmax(torch.zeros(13))

        out = torch.cat((out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8], out[9], out[10], out[11], out[12]), dim=-1) # [24, 160, 768 * 13]
        # out = torch.cat((w_out[0], w_out[1], w_out[2], w_out[3], w_out[4], w_out[5], w_out[6], w_out[7], w_out[8], w_out[9], w_out[10], w_out[11], w_out[12]), dim=-1) # [24, 160, 768 * 13]

        # print("weight", weight)
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