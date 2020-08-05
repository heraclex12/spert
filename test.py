from transformers.src.transformers import *

tokenizer = PhobertTokenizer.from_pretrained("vinai/phobert-base")
# encode = tokenizer.convert_ids_to_tokens([0])
print(tokenizer.pad_token_id)
# print(encode)
# print(tokenizer.tokenize("Ronaldo cũng đang là cầu thủ ghi được nhiều bàn thắng nhất trong lịch sử UEFA Champions League với 126 lần lập công"))
