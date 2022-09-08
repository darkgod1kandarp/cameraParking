from .system import  PARSeq as ModelClass 

import torch
from torch import nn   
import yaml

with open( 'configs/main.yaml', 'r') as f:
        config = yaml.load(f, yaml.Loader)['model']
with open('configs/charset/94_full.yaml', 'r') as f:
    config.update(yaml.load(f, yaml.Loader)['model'])
with open('configs/experiment/parseq.yaml', 'r') as f:
    exp = yaml.load(f, yaml.Loader)
# Apply base model config
model = exp['defaults'][0]['override /model']
with open(f'configs/model/{model}.yaml', 'r') as f:
    config.update(yaml.load(f, yaml.Loader))
# Apply experiment config
if 'model' in exp:
    config.update(exp['model'])



parseq  = ModelClass(**config)
url  = "https://github.com/baudm/parseq/releases/download/v1.0.0/parseq-bb5792a6.pt" # no
checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location='cpu', check_hash=True)
parseq.load_state_dict(checkpoint)
parseq.eval()
