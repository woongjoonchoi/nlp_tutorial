import config
import transformers
import torch.nn as nn


class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        o1  = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        # print(self.bert)
        # print(o2)
        # print(o1['pooler_output'].shape)
        bo = self.bert_drop(o1['pooler_output'])
        output = self.out(bo)
        return output
