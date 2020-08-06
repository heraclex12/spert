import numpy as np

from spert.spPhoBert_model import SpPhoBert
from transformers import *
import torch

test_sentence = """
    Dynamo, tên thật là Steve_Frayne sinh 17 tháng 12 năm 1982 ở Bradford là một nhà ảo_thuật người Anh.
    """
tokenizer = PhobertTokenizer.from_pretrained("vinai/phobert-base")
tokenized_sentence = tokenizer.encode(test_sentence)
input_ids = torch.tensor([tokenized_sentence])

print(input_ids)
# config = PhobertConfig.from_pretrained("data/models/config.json")
model = SpPhoBert.from_pretrained("data/models/pytorch_model.bin",
                                  config="data/models/config.json",
                                  # SpERT model parameters
                                  cls_token=0,
                                  relation_types=2,
                                  entity_types=10,
                                  max_pairs=1000,
                                  prop_drop=0.1,
                                  size_embedding=25,
                                  freeze_transformer=False
                                  )

with torch.no_grad():
    model.eval()
    output = model(input_ids)
    
label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
print(output)

# encode = tokenizer.convert_ids_to_tokens([0])
# print(tokenizer.pad_token_id)
# print(encode)
# print(tokenizer.tokenize("Ronaldo cũng đang là cầu thủ ghi được nhiều bàn thắng nhất trong lịch sử UEFA Champions League với 126 lần lập công"))
