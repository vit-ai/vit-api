import modal 
from fastapi import UploadFile, Response, Form, Request, Body
from typing import Dict
from typing import Annotated

def build_semantic_sam(model_type, ckpt): #todo run in build and cache models
    """
    build model
    """
    from semantic_sam import prepare_image, plot_results, SemanticSamAutomaticMaskGenerator
    from utils.arguments import load_opt_from_config_file
    from semantic_sam.BaseModel import BaseModel
    from semantic_sam import build_model

    cfgs={'T':"/home/appuser/Semantic-SAM/configs/semantic_sam_only_sa-1b_swinT.yaml",
        'L':"/home/appuser/Semantic-SAM/configs/semantic_sam_only_sa-1b_swinL.yaml"}
    sam_cfg=cfgs[model_type]
    opt = load_opt_from_config_file(sam_cfg)
    model_semantic_sam = BaseModel(opt, build_model(opt)).from_pretrained(ckpt).eval().cuda()
    return model_semantic_sam
def startup_SAM():
    import sys
    sys.path.append('/home/appuser/Semantic-SAM/')
    build_semantic_sam(model_type='L', ckpt='/home/appuser/swinl_only_sam_many2many.pth') # model_type: 'L' / 'T', depends on your checkpint

semantic_sam_im = (modal.Image.from_registry('pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel')
                   .apt_install('gcc', 'git', 'libgl1', 'libglib2.0-0', 'wget', 'software-properties-common','libopenmpi-dev', 'ninja-build').conda_install('mpi4py')
                #    .run_commands('wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda-repo-debian11-11-7-local_11.7.0-515.43.04-1_amd64.deb', 'dpkg -i cuda-repo-debian11-11-7-local_11.7.0-515.43.04-1_amd64.deb', 'cp /var/cuda-repo-debian11-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/', 'add-apt-repository contrib').apt_install('cuda')
                   .run_commands('pip3 install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu113', "python -m pip install 'git+https://github.com/MaureenZOU/detectron2-xyz.git'", "pip install git+https://github.com/cocodataset/panopticapi.git", "git clone https://github.com/UX-Decoder/Semantic-SAM /home/appuser/Semantic-SAM")
                   .pip_install("torch","torchvision","opencv-python","pyyaml","json_tricks","yacs","scikit-learn","pandas","timm==0.4.12","numpy==1.23.5","einops","fvcore","transformers==4.19.2","sentencepiece","ftfy","regex","nltk","vision-datasets==0.2.2","pycocotools","diffdist","pyarrow","cityscapesscripts","shapely","scikit-image","mup","gradio==3.35.2","scann","kornia==0.6.4","torchmetrics==0.6.0","progressbar","pillow==9.4.0", "importlib")
                   .run_commands('export CUDA_HOME=/usr/local/cuda-11.6 && export FORCE_CUDA=1 && export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6+PTX" && cd /home/appuser/Semantic-SAM/semantic_sam/body/encoder/ops && sh make.sh')
                   .run_commands('wget -P /home/appuser/ https://github.com/UX-Decoder/Semantic-SAM/releases/download/checkpoint/swinl_only_sam_many2many.pth' ) #replace with local mount
                   .pip_install("addict", "yapf", "supervision==0.6.0")
                   .run_function(startup_SAM, gpu="T4")
                   )


stub = modal.Stub('semantic-sam-inference')

@stub.function(image=semantic_sam_im, gpu='T4')
@modal.web_endpoint(label='semantic-sam-inference', method="POST")
def semantic_sam_inference(uploadfile: UploadFile, levels: list = Form(...), test: bool = Form(...)):
    import sys 
    sys.path.append('/home/appuser/Semantic-SAM/')
    from semantic_sam import prepare_image, plot_results, SemanticSamAutomaticMaskGenerator
    from PIL import Image
    from torchvision import transforms
    import io 
    import numpy as np
    import torch
    import json

    # Load uploaded file and prepare image
    file_bytes = uploadfile.file.read()
    file_stream = io.BytesIO(file_bytes)
    pil_image = Image.open(file_stream).convert('RGB')
    original_size = pil_image.size

    t = []
    t.append(transforms.Resize(640, interpolation=Image.BICUBIC))
    transform1 = transforms.Compose(t)
    image_ori = transform1(pil_image)

    image_ori = np.asarray(image_ori)
    image = torch.from_numpy(image_ori.copy()).permute(2, 0, 1).cuda()

    model = build_semantic_sam(model_type='L', ckpt='/home/appuser/swinl_only_sam_many2many.pth')
    auto_mask_generator = SemanticSamAutomaticMaskGenerator(model, level=[int(level) for level in levels], box_nms_thresh=0.4)
    masks = auto_mask_generator.generate(image)

    # Rescale masks back to original image size
    revert_transform = transforms.Resize(original_size, interpolation=Image.BICUBIC)
    for item in masks:
        if isinstance(item['segmentation'], np.ndarray):
            # Resize segmentation mask to the original size
            mask = Image.fromarray(item['segmentation'])
            mask = revert_transform(mask)
            item['segmentation'] = np.asarray(mask)

            # Convert numpy array to list
            item['segmentation'] = item['segmentation'].tolist()

    json_str = json.dumps(masks)
    print(json_str)
    return Response(content=json_str)


