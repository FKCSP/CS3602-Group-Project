import torch
from dataset.data import LabelConverter, MyDataLoader, MyDataset
from model.decoder import SimpleDecoder
from utils.arguments import arguments
from utils.random import set_random_seed
import json
from transformers import BertModel, BertTokenizer
from utils.output import get_output

test_data_path = 'data/test_unlabelled.json'
output_path = 'data/test.json'

with open(test_data_path, encoding='utf-8') as f:
    test_data = json.load(f)

set_random_seed(114514)


label_converter = LabelConverter('data/ontology.json')
pretrained_model_name = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
bert = BertModel.from_pretrained(pretrained_model_name)

decoder = SimpleDecoder(768, label_converter.num_indexes).to(arguments.device)
check_point = torch.load(open('bert-GRU-final.bin', 'rb'), map_location=arguments.device)
decoder.load_state_dict(check_point['model'])

for i in test_data:
    for j in i:
        text = j['asr_1best']
        model_input = tokenizer(text, return_tensors='pt')
        output = bert(**model_input)[0][0].to(arguments.device).detach()
        output = decoder(output)
        pred = get_output(tokenizer.tokenize(text), output, label_converter)
        j['pred'] = pred

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)
