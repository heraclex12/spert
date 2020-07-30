from transformers_.src.transformers import *

tokenizer = PhobertTokenizer.from_pretrained("vinai/phobert-base")
encode = tokenizer.tokenize("Ronaldo cũng đang là cầu_thủ ghi được nhiều bàn thắng nhất trong lịch_sử UEFA Champions_League với 126 lần lập_công")
print(encode)
print(tokenizer.tokenize("Ronaldo cũng đang là cầu thủ ghi được nhiều bàn thắng nhất trong lịch sử UEFA Champions League với 126 lần lập công"))
