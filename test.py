import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForNextSentencePrediction
import os
import time

print("GPU Available: ", torch.cuda.is_available())
print("GPU Count: ", torch.cuda.device_count())
torch.cuda.set_device(0)

# model_name = 'bert-base-chinese'
model_name = '8a0c070123c1f794c42a29c6904beb7c1b8715741e235bee04aca2c7636fc83f.' \
             '9b42061518a39ca00b8b52059fd2bede8daa613f8a8671500e518a8c29de8c00'
cache_dir = "/home/yuanjun/.pytorch_pretrained_bert"
# model_name = '/Users/quantum/.pytorch_pretrained_bert/' \
#              '8a0c070123c1f794c42a29c6904beb7c1b8715741e235bee04aca2c7636fc83f.' \
#              '9b42061518a39ca00b8b52059fd2bede8daa613f8a8671500e518a8c29de8c00'
# Load pre-trained model tokenizer (vocabulary)

# tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
tokenizer = BertTokenizer.from_pretrained(os.path.join(cache_dir, model_name))

# Tokenized input
tokenized_text = "外面在下雨。出门带伞。"
tokenized_text = tokenizer.tokenize(tokenized_text)
print(tokenized_text)
# Mask a token that we will try to predict back with `BertForMaskedLM`
masked_index = 9
# tokenized_text[masked_index] = '[MASK]'
tokenized_text[masked_index] = '[MASK]'
print(tokenized_text)
assert tokenized_text == ['外', '面', '在', '下', '雨', '。', '出', '门', '带', '[MASK]', '。']

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
segments_ids = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

# Load pre-trained model (weights)
# "bert-base-chinese"
# BertModel.train()
model_name = "42d4a64dda3243ffeca7ec268d5544122e67d9d06b971608796b483925716512." \
             "02ac7d664cff08d793eb00d6aac1d04368a1322435e5fe0a27c70b0b3a85327f"
model = BertModel.from_pretrained(os.path.join(cache_dir, model_name))

model.eval()

# Predict hidden states features for each layer
encoded_layers, _ = model(tokens_tensor, segments_tensors)
# We have a hidden states for each of the 12 layers in model bert-base-uncased
assert len(encoded_layers) == 12

# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained(os.path.join(cache_dir, model_name))
model.cuda()
model.eval()

tokens_tensor_gpu = tokens_tensor.cuda()
segments_tensors_gpu = segments_tensors.cuda()
start = time.time()
count = 0
while True:
    # Predict all tokens
    count += 1
    predictions = model(tokens_tensor_gpu, segments_tensors_gpu)
    # predictions = predictions.cpu()
    # print(predictions.shape)
    # confirm we were able to predict 'henson'
    # predicted_index = torch.argmax(predictions[0, masked_index]).item()
    # predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
    if count % 100 == 0:
        print(count, count/(time.time()-start))
        # print(predicted_token)
    # assert predicted_token[0] == '伞'
    # predicted_index = torch.argmax(predictions[1, masked_index]).item()
    # predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
    # print(predicted_token)
