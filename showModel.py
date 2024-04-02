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

# To run, set the fused_dir, and the val path in the TaskFusionDataset.py
def main(fusion_model_path='./model/Fusion/fusionmodel_final.pth'):
    if not os.path.exists("../model.onnx"):
        device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
        fusionmodel = FusionNet(output=1)
        fusionmodel.load_state_dict(torch.load(fusion_model_path))
        fusionmodel = fusionmodel.to(device)
        dummy_input1 = torch.randn(1, 1, 224, 224)
        dummy_input2 = torch.randn(1, 1, 224, 224)
        fusionmodel.eval()
        torch.onnx.export(fusionmodel, (dummy_input1,dummy_input2), "../model.onnx", verbose=True)
    else:
        import netron
        netron.start("../model.onnx")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SeAFusiuon with pytorch')
    ## model
    parser.add_argument('--model_path', '-M', type=str, default='./model/Fusion/fusionmodel_final.pth')
    ## dataset
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    args = parser.parse_args()
    main(fusion_model_path=args.model_path)
