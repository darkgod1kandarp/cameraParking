from datetime import datetime
from importlib.resources import contents
from typing import List
from fastapi import FastAPI, Body, HTTPException, status, Request, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import uvicorn
from components.ocr import  ocr
from components.license_plate_extraction import numberplate
import numpy as np
import motor.motor_asyncio
from base import numberPLateDetected

lcd = numberplate.Solution()
app  =  FastAPI()   

client = motor.motor_asyncio.AsyncIOMotorClient("mongodb+srv://kandarp:oxjAoe0uwyf58xNo@cluster0.ptbjhv5.mongodb.net/?retryWrites=true&w=majority")

db = client.camuservehicle;


@app.post("/")
def get():
    return JSONResponse(status_code=200 , content = "Nothing")


@app.post("/numberplate/recognition")
async def numberplate_recognition(camData:numberPLateDetected=Body(...)):
    imagearray, location = np.array(camData['imagearray']).astype(np.uint8), 
    val = ocr.OcrThroughArrya(imagearray)
    await db['vechicleplate'].insert_one({"numberplate":val, 'time':datetime.now().timestamp(), 'location':location})
    return  JSONResponse(status_code=200 , content =val )
    
@app.post("/numberplate/detection")  
async  def numberplate_detection(camData:numberPLateDetected =Body(...)):
    camData = jsonable_encoder(camData) 
    imagearray, location = camData['imagearray'],camData['location'] 
    imagearray = np.array(imagearray).astype('float64')
    val = lcd.FetchLicensePlate(imagearray)
    
    val  = val.to_numpy()
    await db['detectedvehicle'].insert_one({"totalNumberPlate":len(val), "location":location, "time":datetime.now().timestamp()})
    
    return JSONResponse(status_code  =  200 , content  = val.tolist())


if __name__=="__main__":
     uvicorn.run("main:app")