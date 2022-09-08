from PIL import Image
import numpy
from components.ocr.config import parseq, img_transform
import torch as torch
import cv2
import numpy as np
from scipy.ndimage import interpolation as inter


class ImageToOcr():
    def __init__(self) -> None:
        pass

    def OcrThroughArrya(self, imgArray):
        # imgArray = torch.from_numpy(imgArray)
        imgArray = Image.fromarray(imgArray)
        imgArray = img_transform(imgArray).unsqueeze(0)

        logits = parseq(imgArray)
        logits.shape  # torch.Size([1, 26, 95]), 94 characters + [EOS] symbol

        # Greedy decoding
        pred = logits.softmax(-1)
        label, confidence = parseq.tokenizer.decode(pred)
        return label[0]

    def OcrThroghPath(self, path):

        img = Image.open(path).convert('RGB')

        # Preprocess. Model expects a batch of images with shape: (B, C, H, W)
        img = img_transform(img).unsqueeze(0)
        print(img.shape)
        logits = parseq(img)
        logits.shape  # torch.Size([1, 26, 95]), 94 characters + [EOS] symbol

        # Greedy decoding
        pred = logits.softmax(-1)
        label, confidence = parseq.tokenizer.decode(pred)
        return label[0]


class Load_torch_plate:
    def __init__(self) -> None:
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='components/license_plate_extraction/last.pt', force_reload=True)
        self.number_plate = None
        self.ocr = ImageToOcr()

    def FetchLicensePlate(self, img_path: str):
        results = self.model(img_path)
        df = results.pandas().xyxy[0]
        results.show()
        number_plate = df[df['name'] == "license"]
       
        self.number_plate = number_plate if len(number_plate) > 0 else False
        
    def ReadImage(self, img_path: str):
        img = cv2.imread(img_path)
        y, x =  img.shape[0], img.shape[1]
        img = cv2.resize(img , (y*2 , x*2),cv2.INTER_CUBIC)
        return img
        
        

    def CropImage(self, img_path):

        
        img = self.ReadImage(img_path)
        self.FetchLicensePlate(img)
       
        if type(self.number_plate) is bool:
            return False
        
        self._ListOfNumberPlate = []
        
        # print(len(self.))
        for idx in range(len(self.number_plate)):

            x1, y1, x2, y2 = int(self.number_plate['xmin'][idx]), int(self.number_plate['ymin'][idx]), int(
                self.number_plate['xmax'][idx]), int(self.number_plate['ymax'][idx])
            print(x1 , x2 , y1 , y2 )
            val_img  = img[y1:y2, x1:x2]
            
            
            val = self.RecognizeNumberPlate(val_img)
            
            cv2.imshow(val, val_img)

            self._ListOfNumberPlate.append(img[y1:y2, x1:x2])
            cv2.waitKey(0)

    def RecognizeNumberPlate(self, array):
        # op,array = self.ocr.correct_skew(array)
        # self.ocr.line_segmentation(array)
        return self.ocr.OcrThroughArrya(array)







license_plate = Load_torch_plate()

file = '90_jpg.rf.9c0bae6213e2ca535983329a88336149.jpg'
root_dir = f"/home/om/ml/Number-plate-1/test/images/{file}"




license_plate.CropImage(root_dir)

ocr = ImageToOcr() 
print(ocr.OcrThroghPath(root_dir))

# image = cv2.imread(root_dir)
# angle, rotated = correct_skew(image)
# print(angle)
# cv2.imshow('rotated', rotated)
# cv2.imwrite('rotated.png', rotated)
# cv2.waitKey()
