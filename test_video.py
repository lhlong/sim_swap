import cv2
import torch
import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from insightface_func.face_detect_crop_multi import Face_detect_crop
from util.videoswap import video_swap
from util.add_watermark import watermark_image
import os


transformer = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

detransformer = transforms.Compose([
        transforms.Normalize([0, 0, 0], [1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1])
    ])

import time

t0 = time.time()

opt = TestOptions()
opt.initialize()
opt.parser.add_argument('-f') ## dummy arg to avoid bug
opt = opt.parse()
# opt.pic_a_path = './demo_file/Iron_man.jpg' ## or replace it with image from your own google drive
#opt.pic_a_path = './demo_file/multispecific/DST_01.jpg'
# opt.video_path = './demo_file/multi_people_1080p.mp4' ## or replace it with video from your own google drive
# opt.video_path = './demo_file/video4.mp4' ## or replace it with video from your own google drive

# opt.video_path = './demo_file/video4.mp4' ## or replace it with video from your own google drive
# opt.output_path = './output/demo_video4.mp4'


input_dir = "./demo_file"
output_dir = "./output"
image_filename = 'Iron_man.jpg'
video_filename = "video4.mp4"
opt.pic_a_path = os.path.join(input_dir, image_filename) ## or replace it with image from your own google drive
opt.video_path = os.path.join(input_dir, video_filename) ## or replace it with video from your own google drive
opt.output_path = os.path.join(output_dir, video_filename.split('.')[0] + "_" + image_filename.split('.')[0] + ".mp4")

print(opt.pic_a_path)
print(opt.video_path)
print(opt.output_path)

opt.temp_path = './tmp'
opt.Arc_path = './arcface_model/arcface_checkpoint.tar'
opt.isTrain = False
# opt.use_mask = True  ## new feature up-to-date
opt.use_mask = False  ## new feature up-to-date


crop_size = opt.crop_size

torch.nn.Module.dump_patches = True
model = create_model(opt)
model.eval()

app = Face_detect_crop(name='antelope', root='./insightface_func/models')
app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640))

with torch.no_grad():
    pic_a = opt.pic_a_path
    # img_a = Image.open(pic_a).convert('RGB')
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
    latend_id = latend_id.detach().to('cpu')
    latend_id = latend_id/np.linalg.norm(latend_id,axis=1,keepdims=True)
    latend_id = latend_id.to('cuda')

    video_swap(opt.video_path, latend_id, model, app, opt.output_path, temp_results_dir=opt.temp_path, use_mask=opt.use_mask)
    
print("Total time: ", time.time()-t0)

