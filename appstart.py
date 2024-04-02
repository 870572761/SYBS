# coding:utf-8
import os
import argparse
from utils import *
import torch
from torch.utils.data import DataLoader
from datasets import Fusion_dataset
from FusionNet import FusionNet
from tqdm import tqdm
import gradio as gr
from pathlib import Path
import torchvision.transforms.functional as TF
import cv2

examples = [
            ["Picture","test_imgs/vi/17.png","test_imgs/ir/17.png",None,None],
            ["Picture","test_imgs/vi/21.png","test_imgs/ir/21.png",None,None],
            ["Picture","test_imgs/vi/36.png","test_imgs/ir/36.png",None,None],
            ["Picture","test_imgs/vi/00633D.png","test_imgs/ir/00633D.png",None,None],
            ["Video",None,None,"../data/rainlightvi.mp4","../data/rainlightir.mp4"]
            ]
class App:
    def __init__(self,device,fusionmodel):
        self.device = device
        self.fusionmodel = fusionmodel
    
    def Image2tensor(self,img,type):
        if type == 1:
            img = img.convert('RGB')
        elif type == 2:
            img = img.convert('L')
        img = TF.to_tensor(img)
        img = img.unsqueeze(0)
        return img
    
    def forward(self,img_vis,img_ir):
        with torch.no_grad():
            img_vis = img_vis.to(self.device)
            img_ir = img_ir.to(self.device)
            vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(img_vis)
            vi_Y = vi_Y.to(self.device)
            vi_Cb = vi_Cb.to(self.device)
            vi_Cr = vi_Cr.to(self.device)
            fused_img = self.fusionmodel(vi_Y, img_ir)
            fused_img = YCbCr2RGB(fused_img, vi_Cb, vi_Cr)
            img = tensor2img(fused_img[0], is_norm=True)
            img = Image.fromarray(img)
        return img
    
    def save_frames_as_video(res, output_path = Path("./Result.mp4"), fps=30.0):
        height, width, _ = res[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for frame in res:
            video_writer.write(frame)
        video_writer.release()
    
    def forwardvideo(self,vidvi,vidir,respath = Path("./Result.mp4")):
        vi,ir = cv2.VideoCapture(vidvi),cv2.VideoCapture(vidir)
        fps = min(vi.get(cv2.CAP_PROP_FPS),ir.get(cv2.CAP_PROP_FPS))
        res = []
        while True:
            successvi, imgvi = vi.read()
            successir, imgir = ir.read()
            if not successvi & successir:
                break
            imgvi, imgir = self.Image2tensor(imgvi,3), self.Image2tensor(imgir,4)
            resimg = self.forward(imgvi, imgir)
            res.append(resimg)
        self.save_frames_as_video(res, respath, fps)
    
    def predict(self,task,imgvi,imgir,vidvi,vidir):
        if task == 0:
            imgvi, imgir = self.Image2tensor(imgvi,1), self.Image2tensor(imgir,2)
            output = self.forward(imgvi,imgir)
            return output,None
        if task == 1:
            respath = Path("./Result.mp4")
            self.forwardvideo(vidvi, vidir, respath)
            return None,respath
# To run, set the fused_dir, and the val path in the TaskFusionDataset.py
def main(fusion_model_path='./model/Fusion/fusionmodel_final.pth'):
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    fusionmodel = FusionNet(output=1)
    fusionmodel.load_state_dict(torch.load(fusion_model_path))
    fusionmodel = fusionmodel.to(device)
    app = App(device,fusionmodel)
    gr.Interface(
             fn=app.predict,
             inputs=[gr.Dropdown(["Picture","Video"],label="Task",type='index'),gr.Image(type="pil"),gr.Image(type="pil"),gr.Video(),gr.Video()],
             outputs=[gr.Image(type="pil"),gr.Video()],
             examples=examples
             ).launch(share = True)
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SeAFusiuon with pytorch')
    ## model
    parser.add_argument('--model_path', '-M', type=str, default='./model/Fusion/fusionmodel_final.pth')
    ## dataset
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    args = parser.parse_args()
    main(fusion_model_path=args.model_path)
