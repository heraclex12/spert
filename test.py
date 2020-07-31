from transformers_.src.transformers import *

tokenizer = PhobertTokenizer.from_pretrained("vinai/phobert-base")
encode = tokenizer.tokenize('<s>')
print(tokenizer.cls_token_id)
# print(tokenizer.tokenize("Ronaldo cũng đang là cầu thủ ghi được nhiều bàn thắng nhất trong lịch sử UEFA Champions League với 126 lần lập công"))
