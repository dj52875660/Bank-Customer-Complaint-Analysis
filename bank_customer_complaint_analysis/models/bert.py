import torch
import torch.nn as nn

# from transformers import BertForSequenceClassification
from transformers import BertModel


class TextClassificationModel(nn.Module):
    def __init__(self, num_labels):
        super(TextClassificationModel, self).__init__()
        self.num_labels = num_labels
        # self.bert = BertForSequenceClassification.from_pretrained(
        #     "bert-base-uncased", num_labels=num_labels
        # )
        # self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # logits = outputs.logits
        # logits = self.dropout(outputs.logits)  # 添加 Dropout 層

        pooled_output = outputs.last_hidden_state[:, 0, :]  # 取出 [CLS] token 的輸出
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits
