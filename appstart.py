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
from model_TII import BiSeNet

from ultralytics import YOLO

examples = [
            ["Detect",["RGB","Infread"],"Picture","test_imgs/vi/17.png","test_imgs/ir/17.png",None,None],
            [["Detect","Count"],["RGB","Infread"],"Picture","test_imgs/vi/21.png","test_imgs/ir/21.png",None,None],
            [["Detect","Count"],["RGB"],"Picture","test_imgs/vi/21.png","test_imgs/ir/21.png",None,None],
            ["Detect",["RGB","Infread"],"Picture","test_imgs/vi/36.png","test_imgs/ir/36.png",None,None],
            ["Detect",["RGB","Infread"],"Picture","test_imgs/vi/00633D.png","test_imgs/ir/00633D.png",None,None],
            [["Detect","Seg"],["RGB","Infread"],"Picture","test_imgs/vi/00633D.png","test_imgs/ir/00633D.png",None,None],
            [["Detect","Seg"],["RGB"],"Picture","test_imgs/vi/00633D.png",None,None,None],
            # [None,["RGB","Infread"],"Video",None,None,"../data/rainlightvi.mp4","../data/rainlightir.mp4"],
            ["Detect",["RGB","Infread"],"Video",None,None,"../data/rainlightvi.mp4","../data/rainlightir.mp4"],
            ["Detect",["RGB"],"Video",None,None,"../data/rainlightvi.mp4","../data/rainlightir.mp4"]
            ]

# 定义自定义调色板
palette = np.array([[0, 0, 0],   # 黑色 背景
                    [0, 0, 0],    # 红色 
                    [0, 255, 0],   # 绿色 
                    [0, 0, 255],   # 蓝色 人
                    [255, 255, 0], # 黄色
                    [255, 0, 255], # 紫色
                    [0, 255, 255], # 青色
                    [128, 0, 0],   # 暗红色
                    [0, 128, 0],   # 暗绿色
                    [0, 0, 128],   # 暗蓝色
                    [128, 128, 0], # 暗黄色
                    [128, 0, 128], # 暗紫色
                    [0, 128, 128],]# 暗青色
                   )

inputs = [
    gr.CheckboxGroup(["Detect","Seg","Count"],label="Vision task", info="What vision task do you want to do ?"),
    gr.CheckboxGroup(["RGB","Infread"],label="Vision mode", info="What mode do you wanna add?"),
    gr.Dropdown(["Picture","Video"],label="Task",type='index'),
    gr.Image(type="pil"),
    gr.Image(type="pil"),
    gr.Video(),
    gr.Video(),
]
class App:
    def __init__(self,device,fusionmodel,detectmodel=None,net=None,palette=None):
        self.device = device
        self.fusionmodel = fusionmodel
        self.detectmodel = detectmodel
        self.net = net
        self.palette = palette

    
    def Image2tensor(self,img,type):
        if type == 1:
            img = img.convert('RGB')
        elif type == 2:
            img = img.convert('L')
        img = TF.to_tensor(img)
        img = img.unsqueeze(0)
        return img
    
    def change2textres(self,names=None,ncls=None):
        # 统计每个类别的检测数量
        class_counts = torch.zeros(len(names), dtype=torch.int)
        predicted_classes = torch.round(ncls).long()
        for class_idx in predicted_classes:
            class_counts[class_idx] += 1
        result = {names[i]: int(class_counts[i]) for i in range(len(names)) if class_counts[i] > 0}
        return result
    


    def forward(self,img_vis,img_ir,task,format="IMG",mode=["RGB","Infread"]):
        rescount = None
        # img = None
        if "RGB" in mode and "Infread" in mode:
            with torch.no_grad():
                img_vis = img_vis.to(self.device)
                img_ir = img_ir.to(self.device)
                vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(img_vis)
                vi_Y = vi_Y.to(self.device)
                vi_Cb = vi_Cb.to(self.device)
                vi_Cr = vi_Cr.to(self.device)
                fused_img = self.fusionmodel(vi_Y, img_ir)
                fused_img = YCbCr2RGB(fused_img, vi_Cb, vi_Cr)
                img = fused_img
                ori_img = tensor2img(fused_img[0], is_norm=True)
        elif  "RGB" in mode:
            img = img_vis
            ori_img = np.array(img)#ori为将上色图像
        elif "Infread" in mode:
            img = img_ir
            ori_img = np.array(img)#ori为将上色图像
        #img永远都是PIL格式或者tensor格式
        if "Detect" in task:
            result = self.detectmodel(ori_img)
            annotated_frame = result[0].plot()
            ori_img = annotated_frame
        if "Seg" in task:
            # if isinstance(img, np.ndarray):#如果是融合输出
            #     img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
            if not isinstance(img, torch.Tensor):#如果不是tensor格式，为普通输出
                img = self.Image2tensor(img,1)
                img = img.to(self.device)
            out, _ = self.net(img)
            out = out.clamp(min=0).floor()
            out = np.argmax(out.squeeze().permute(1, 2, 0).detach().cpu().numpy(), axis=-1)
            palette = self.palette
            overlay_mask = np.zeros_like(ori_img)
            for i in range(out.shape[0]):
                for j in range(out.shape[1]):
                    overlay_mask[i, j] = palette[out[i, j]]
            alpha = 0.3
            resultseg = cv2.addWeighted(ori_img, 1-alpha, overlay_mask, alpha, 1)
            ori_img = resultseg
        if "Count" in task:
            rescount = self.change2textres(result[0].names,result[0].boxes.cls)
        if format == "IMG":
            ori_img = Image.fromarray(ori_img)
        return ori_img,rescount
    
    def save_frames_as_video(self,res, output_path = Path("./Result.avi"), fps=30.0):
        from moviepy.editor import ImageSequenceClip
        clip = ImageSequenceClip(res, fps=fps)
        clip.write_videofile(str(output_path))
        return
    
    def forwardvideo(self,vidvi,vidir,task,respath = Path("./Result.avi"),mode=["RGB","Infread"]):
        vi,ir = cv2.VideoCapture(vidvi),cv2.VideoCapture(vidir)
        fps = min(vi.get(cv2.CAP_PROP_FPS),ir.get(cv2.CAP_PROP_FPS))
        total_frames = max(vi.get(cv2.CAP_PROP_FRAME_COUNT),ir.get(cv2.CAP_PROP_FRAME_COUNT))
        # total_time = total_frames / fps  # 视频总时间长度（秒）
        print("Total",total_frames)
        res = []
        i = -1 
        while True:
            i += 1
            # print(i)
            successvi, imgvi = vi.read()
            successir, imgir = ir.read()
            if not (successvi & successir):
                print(False)
                break
            if i % 5 != 0:
                continue
            imgir = cv2.cvtColor(imgir, cv2.COLOR_BGR2GRAY)
            imgvi, imgir = self.Image2tensor(imgvi,3), self.Image2tensor(imgir,4)
            resimg,countres = self.forward(imgvi, imgir, task, format="NP", mode=mode)
            res.append(resimg)
            
        self.save_frames_as_video(res, respath, fps)
    
    def predict(self,task,mode,type,imgvi,imgir,vidvi,vidir):
        if type == 0:
            if "RGB" in mode and "Infread" in mode:
                imgvi = self.Image2tensor(imgvi,1)
                imgir = self.Image2tensor(imgir,2)
            output,countres = self.forward(imgvi,imgir,task,mode=mode)
            print(countres)
            return output,None,countres
        if type == 1:
            # return None,vidvi,None
            respath = Path("./Result.mp4")
            self.forwardvideo(vidvi, vidir,task,respath,mode)
            countres = None
            return None,str(respath),countres
# To run, set the fused_dir, and the val path in the TaskFusionDataset.py
def main(fusion_model_path='./model/Fusion/fusionmodel_final.pth',detectmodel_path="../weights/yolov8s.pt"):
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    fusionmodel = FusionNet(output=1)
    fusionmodel.load_state_dict(torch.load(fusion_model_path))
    fusionmodel = fusionmodel.to(device)
    load_path = './model/Fusion/model_final.pth'
    net = BiSeNet(n_classes=9)
    net.load_state_dict(torch.load(load_path))
    net.eval()
    detectmodel = YOLO(detectmodel_path)
    # detectmodel = YOLO("../weights/yolomhead.pt")
    app = App(device,fusionmodel,detectmodel,net,palette)
    gr.Interface(
            fn=app.predict,
            inputs=inputs,
            outputs=[gr.Image(type="pil"),gr.Video(),gr.Textbox(label="Count")],
            examples=examples
            ).launch(share = True)
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SeAFusiuon with pytorch')
    ## model
    parser.add_argument('--model_path', '-M', type=str, default='./model/Fusion/fusionmodel_final.pth')
    parser.add_argument('--detectmodel_path', '-D', type=str, default='../weights/yolov8s.pt')
    ## dataset
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    args = parser.parse_args()
    main(fusion_model_path=args.model_path,detectmodel_path=args.detectmodel_path)
