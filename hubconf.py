# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
PyTorch Hub models https://pytorch.org/hub/ultralytics_yolov5

Usage:
    import torch
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    model = torch.hub.load('ultralytics/yolov5:master', 'custom', 'path/to/yolov5s.onnx')  # custom model from branch
"""


from pathlib import Path
from models.yolo import  DetectionModel

from utils.torch_utils import select_device


def _create(name, channels=3, classes=80,  device=None):
    
    

    
    name = Path(name)
    path = name.with_suffix('.pt') if name.suffix == '' and not name.is_dir() else name  # checkpoint path
    try:
        device = select_device(device)
        # arbitrary model
       
        cfg = '/home/om/ml/softwatre engineering/ModelServer/models/yolov5s.yaml'  # model.yaml path
        model = DetectionModel(cfg, channels, classes)  # create model
              # reset to default
        return model.to(device)

    except Exception as e:
        print(e)
        help_url = 'https://github.com/ultralytics/yolov5/issues/36'
        s = f'{e}. Cache may be out of date, try `force_reload=True` or see {help_url} for help.'
        raise Exception(s) from e

model  = _create(name = 'components/license_plate_extraction/last.pt')
file = '90_jpg.rf.9c0bae6213e2ca535983329a88336149.jpg'
root_dir = f"/home/om/ml/Number-plate-1/test/images/{file}"
result  =  model(root_dir)
result.show()




