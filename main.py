from fastapi import FastAPI, status, Request, HTTPException, Depends, UploadFile, File
import json
from typing import List
from utils.constants import AUTH_USER, AUTH_PWD, save_path
import datetime
from fastapi.openapi.utils import get_openapi
from starlette.responses import RedirectResponse, Response, JSONResponse
from pydantic import BaseModel, Field
import secrets
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.openapi.docs import get_swagger_ui_html
import os
import cv2
import torch
import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from insightface_func.face_detect_crop_single import Face_detect_crop
from util.videoswap import video_swap
import os

opt = TestOptions().parse()
opt = TestOptions()

opt.initialize()
opt.parser.add_argument('-f') ## dummy arg to avoid bug
opt = opt.parse()

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

security = HTTPBasic()

class video_store(BaseModel):
    video_id: str
    video_path: str

transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# class video_store(BaseModel):
#     video_id: str
#     video_path: str
#     image: File 

def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, AUTH_USER)
    correct_password = secrets.compare_digest(credentials.password, AUTH_PWD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


@app.get("/openapi.json")
async def get_open_api_endpoint(auth=Depends(get_current_username)):
    return JSONResponse(get_openapi(title="FD-API", version=1, routes=app.routes))


@app.get("/docs")
async def get_documentation(auth=Depends(get_current_username)):
    return get_swagger_ui_html(openapi_url="/openapi.json", title="docs")


def validate(date_text):
    try:
        if date_text != datetime.datetime.strptime(date_text, "%Y-%m-%d").strftime('%Y-%m-%d'):
            raise ValueError
        return True
    except ValueError:
        return False


@app.get("/")
def read_root():
    return {"FD": "Version:0.0.1"}



@app.post("/video/cache")
def store_video(video: video_store,auth=Depends(get_current_username)):
    try:
        # Check whether the specified path exists or not
        isExist = os.path.exists(save_path)
        if not isExist:
        # Create a new directory because it does not exist
            os.makedirs(save_path)
        #Todo: check dinh dang video
        file_name = save_path+"/"+video.video_id+".mp4"
        #Todo: download file S3 or ?
        # result =  get_similar_word_by_wordids(word.word_ids,word.threshold,word.action,word.score_type,word.word_type, word.distance_type)
        return {"status":200}
    except Exception as e:
        return {"status": 400}


@app.post("/video/swap")
def submit(video: video_store = Depends(), files: List[UploadFile] = File(...)):
    received_data= video.dict()
    video_path = save_path + received_data["video_id"]+".mp4"
    opt.pic_a_path = files[0] ## or replace it with image from your own google drive
    opt.video_path = video_path ## or replace it with video from your own google drive
    opt.output_path = './output/demo2.mp4'
    crop_size = opt.crop_size

    torch.nn.Module.dump_patches = True
    if crop_size == 512:
        opt.which_epoch = 550000
        opt.name = '512'
        mode = 'ffhq'
    else:
        mode = 'None'
    model = create_model(opt)
    model.eval()


    app = Face_detect_crop(name='buffalo_sc', root='./insightface_func/models')
    app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640),mode=mode)
    with torch.no_grad():
        pic_a = opt.pic_a_path
        img_a_whole = cv2.imread(pic_a)
        img_a_align_crop, _ = app.get(img_a_whole,crop_size)
        img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0],cv2.COLOR_BGR2RGB)) 
        img_a = transformer_Arcface(img_a_align_crop_pil)
        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

        # convert numpy to tensor
        img_id = img_id.cuda()

        #create latent id
        img_id_downsample = F.interpolate(img_id, size=(112,112))
        latend_id = model.netArc(img_id_downsample)
        latend_id = F.normalize(latend_id, p=2, dim=1)

        video_swap(opt.video_path, latend_id, model, app, opt.output_path,temp_results_dir=opt.temp_path,\
            use_mask=opt.use_mask,crop_size=crop_size)
    return {"JSON Payload ": received_data, "Uploaded Filenames": [file.filename for file in files]}

