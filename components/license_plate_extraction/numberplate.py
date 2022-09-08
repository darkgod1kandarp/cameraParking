import torch as torch
from typing import List
import numpy as np
torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

class Solution:
    def __init__(self) -> None:
    
        self.model   = torch.hub.load('ultralytics/yolov5', 'custom', path='components/license_plate_extraction/last.pt')

    def FetchLicensePlate(self,  imgarray:List ):
        
        results = self.model(imgarray)

        df = results.pandas().xyxy[0]
     
        number_plate = df[df['name'] == "license"]
        return  number_plate if len(number_plate) > 0 else False