import requests  # noqa
import cv2  # noqa
import json


file = '75_jpg.rf.51eef46293f592c2aa71313b03a7d611.jpg'
root_dir = f"/home/om/ml/Number-plate-1/valid/images/{file}"
img = cv2.imread(root_dir)
y, x =  img.shape[0], img.shape[1]
img = cv2.resize(img , (y*2 , x*2),cv2.INTER_CUBIC)


response = requests.post("http://0.0.0.0:4557/numberplate/detection", json ={'imagearray':img.tolist(), 'location':'AU/navrangpura/commerce 6 rasta'}) 
cord_resp = response.json()
print(cord_resp)
for cords in cord_resp: 
    x1 , y1 , x2  ,y2= list(map(int , cords[0:4]))
    img1 =  img[y1:y2 , x1:x2]
   
    response  =  requests.post("http://0.0.0.0:4557/numberplate/recognition" , json = {'data':img1.tolist()})
    print(response.json())