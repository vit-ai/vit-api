import modal 
from fastapi import UploadFile, Response
stub = modal.Stub('clip-inference')

def initialize_clip(): # add parameter for specific model
    import open_clip
    model_clip, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', cache_dir="/content/model")
    model_clip.eval()
    model_clip = model_clip.cuda()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    return model_clip, preprocess, tokenizer

def get_clip_label(model_clip, preprocess, tokenizer, pil_image, label_list, n=3):
    
    import torch

    if n > len(label_list): 
        n = len(label_list)
    image = preprocess(pil_image).cuda().unsqueeze(0)
    text = tokenizer(label_list).cuda()
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model_clip.encode_image(image)
        text_features = model_clip.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        # Get the top n probabilities and corresponding indices
        top_probs, top_indices = torch.topk(text_probs, n, dim=-1)
        # Retrieve the labels and probabilities
        top_labels = [label_list[index] for index in top_indices[0]]
        top_probabilities = top_probs[0].tolist()
    return list(zip(top_labels, top_probabilities))

build_image = (modal.Image.from_dockerfile('/Users/liammckenna/vit-ai/api/vit-api/gpu_api/Dockerfile').pip_install('open_clip_torch', 'opencv-python', 'numpy')
         )

@stub.function(image=build_image, gpu='T4')
@modal.web_endpoint(label='clip-inference', method="POST")
def clip_inference(uploadfile: UploadFile, label_list: UploadFile):
    import io
    import cv2
    import numpy as np
    #convert cv2 image to PIL image
    from PIL import Image
    import json
    

    file_bytes = uploadfile.file.read()
    file_stream = io.BytesIO(file_bytes)

    label_list = json.loads(label_list.file.read())
    image_bgr = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    model_clip, preprocess, tokenizer = initialize_clip()
    clip_labels = get_clip_label(model_clip, preprocess, tokenizer, pil_image, label_list)
    return Response(content=json.dumps(clip_labels))