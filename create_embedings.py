import numpy as np
import cv2 as cv
import argparse
import torch
import onnxruntime
import glob
import os
import shutil
from mtcnn import MTCNN
from PIL import Image
def calculate_distance(feat1, feat2, metric):

    if (metric == "cosine"):
        # feat1 = feat1/np.linalg.norm(feat1 , axis=-1 , keepdims=True)
        # feat2 = feat2/np.linalg.norm(feat2 , axis=-1 ,  keepdims=True)
        dist_m = 1 - np.matmul(feat1 , feat2.T)
        return dist_m[0,0]



def preprocess(image , img_size):
    
    img = cv.resize(image , (img_size[0] , img_size[1]))
    img = img.astype("float32").transpose(2 , 0 , 1)[np.newaxis]/255.0
    return img 


def postprocess(feat):
    feature = feat/(np.linalg.norm(feat , axis=-1 , keepdims=True)+1e-5)
    return feature 



def load_img(img_path ):
    img = cv.imread(img_path)
    image = cv.cvtColor(img , cv.COLOR_BGR2RGB)
    return img 


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gallery" , type=str, default= "/home/altex/Desktop/PersonRegistorySampleVideos/SharifCameras/21/1/2024-06-02_15_20_16/CameraEvent_2024-06-02 15:20:15#1091277/0/"),
    parser.add_argument("--model" , type=str,default="models/model_ir_se50.onnx" ),
    parser.add_argument("--runtime" , type=str,default="onnx" ),
    parser.add_argument("--result", type=str,default="outputs_onnx" )

    return parser

def load_model(model_path , runtime):
    if (runtime == "torchscript"):
        script_model = torch.jit.trace(model_path,torch.rand(1,3,112,112))
        script_model.save(model_path + "torchscript")
        model = torch.jit.load(model_path + "torchscript")
        return model
    
    if (runtime == "onnx"):
        session = onnxruntime.InferenceSession(model_path,providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        return lambda x : session.run(None , {input_name : x})[0]
    

if __name__=="__main__":
    args = parser().parse_args()
    model_path = args.model

    mtcnn = MTCNN()
    threshold = 0.35
    model = load_model(model_path , args.runtime)

    features = []
    files = []
    for img_path in glob.glob(args.gallery+'/*.jpg'): 
        if True:
            try:
                image = Image.open(img_path)
                bboxes, faces = mtcnn.align_multi(image, 10, 16,)
                # k_img = load_img(img_path , [112,112]) 
                j = 0
                for face,bbox in zip(faces,bboxes):
                    bbox = np.int32(bbox)
                    oface = np.array(image)[bbox[1]:bbox[3],bbox[0]:bbox[2]]
                    face = np.array(face)
                    face_path = "/home/altex/Codes/FaceRecognition_Detection/InsightFace_Pytorch/data/"+str(j)+"_"+img_path.split('/')[-1]
                    cv.imwrite(face_path,cv.cvtColor(face,cv.COLOR_RGB2BGR) )
                    j+=1
                    face = preprocess(face,[112,112])
                    k_feat = model(face) 
                # k_feat = postprocess(k_feat)  
                    features.append(k_feat)
                    files.append(face_path)
            except Exception as e:
                print(e)
                continue
        else:
            face = load_img(img_path)
            # cv.imwrite("data/"+img_path.split('/')[-1],face)
            face = preprocess(face,[112,112])
            k_feat = model(face) 
        # k_feat = postprocess(k_feat)  
            features.append(k_feat)
            files.append(img_path)


    output_dir = args.result
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    np.save(output_dir+'/embeddings',features)
    np.save(output_dir+'/paths',files)