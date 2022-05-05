from argparse import ArgumentParser
import os
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models
import numpy as np
import torch
from segmentation_model import evaluator

parser = ArgumentParser()
parser.add_argument("img_path", help="The relative path for the image that you want to remove the fence for", type=str)
parser.add_argument("-o", "--output", help="The relative path for the mask to be exported to", type=str)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)

def main():
    args = parser.parse_args()

    # Check if valid path
    if not os.path.isfile(args.img_path):
        print(f"{args.img_path} is not a valid file path!")
        return

    # Check if valid file 
    if not args.img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        print(f"Please make sure {args.img_path} is a png, jpg, or jpeg image!")
        return
    
    # Check if custom output path given and valid
    if args.output and os.path.isdir(args.output):
        out_path = os.path.join(args.output, os.path.basename(args.img_path)).rsplit('.', 1)[0]
    else:
        out_path = args.img_path.rsplit('.', 1)[0]


    try:
        img = Image.open(args.img_path).convert("RGB")
        
        # Perform transforms on the input image
        toTensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if img.size != (256,256):
            resize = transforms.Resize((256,256))
            img_transforms = transforms.Compose((resize, toTensor, normalize))
        else:
            img_transforms = transforms.Compose((toTensor, normalize))
        inp = torch.unsqueeze(img_transforms(img), 0)

        # Retrieve the trained model and get the mask
        img_seg_model = get_model(1)
        mask = evaluator.get_inference_output(img_seg_model, inp, device)
        mask = torch.where(mask > 0, 1, 0).cpu().reshape(1, 256, 256, -1)[0]

        # Save the binary mask as a png
        mask_path = out_path + '-mask.png'
        mask_img = Image.fromarray(np.squeeze(np.asarray(mask*255, dtype=np.uint8), 2))
        mask_img.save(mask_path, format='PNG')
        print(f"Mask has been saved to {mask_path}")

    except Exception as e:
        print(f"Exception: {str(e)}")
        return

def get_model(n):
    if n==1:
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(2048, 65536)
        model.load_state_dict(torch.load('saved_models/resnet50-v2.model', map_location=torch.device(device)))
        return model
    else:
        raise Exception("function=get_model: Invalid model number")

if __name__ == '__main__':
    main()