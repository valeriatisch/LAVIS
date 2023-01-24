import os
import requests
import torch
import numpy as np
import json
import csv
import random
from PIL import Image, ImageOps
from lavis.models import load_model_and_preprocess
from matplotlib import pyplot as plt, image as pltimg
from lavis.common.gradcam import getAttMap
from lavis.models.blip_models.blip_image_text_matching import compute_gradcam
from pathlib import Path

img_path = Path('/hpi/fs00/share/fg-naumann/seminar-ws22-tagging-captioning-art/artpedia-data/images/')
json_path = Path('/hpi/fs00/home/elena.gensch/data/artpedia/artpedia.json')
output_path = Path('/hpi/fs00/home/elena.gensch/dev/output/')

    
def img_name_from(url: str):
    return url.split("/")[-1].split(".")[0] + '.png'


def load_image(img_url: str):
    return Image.open(requests.get(img_url, stream=True).raw).convert("RGB")


def predict_caption(raw_image, captioner, force_words = None):
    model, vis_processors, text_processors = captioner
    force_words_ids = None
    if force_words:
        force_words_ids = model.tokenizer(force_words, add_special_tokens=False).input_ids
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)    
    pred_captions = model.generate({"image": image}, use_nucleus_sampling=False, num_captions=3, num_beams = 3, force_words_ids = force_words_ids, max_length = 70)

    return pred_captions


def normalize(raw_image, gray_scale: bool = True):
    dst_w = 720
    w, h = raw_image.size
    scaling_factor = dst_w / w

    if gray_scale:
        gray = ImageOps.grayscale(raw_image)

    resized_img = gray.resize((int(w * scaling_factor), int(h * scaling_factor)))
    norm_img = np.float32(resized_img) / 255
    norm_img = np.stack((norm_img,)*3, axis=-1)
    return norm_img


def visualize_attention(raw_image, text_matcher, caption):
    model, vis_processors, text_processors = text_matcher
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    norm_img = normalize(raw_image)
    txt = text_processors["eval"](caption)
    txt_tokens = model.tokenizer(txt, return_tensors="pt").to(device)
    gradcam, _ = compute_gradcam(model, image, txt, txt_tokens, block_num=7)
    avg_gradcam = getAttMap(norm_img, gradcam[0][1], blur=True)

    return avg_gradcam


def load_json_labels(path: str):
    with open(path, "r") as json_file:
        data = json.load(json_file)
    images_labels = {}
    for id, sample in data.items():
        images_labels[img_name_from(sample["img_url"])] = sample["visual_sentences"]
    return images_labels


def log_captions(captions: list[str], file_name: str):
    entries = [file_name]
    entries.extend(captions)
    with open(output_path / 'output.csv', 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(entries)



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # loads BLIP caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
    captioner = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)
    text_matcher = load_model_and_preprocess(name="blip_image_text_matching", model_type="large", device=device, is_eval=True)

    images_labels = load_json_labels(json_path)
    for file in random.sample(list(img_path.glob(pattern = '**/*.png')), 10):
        raw_image = Image.open(file)

        captions = predict_caption(raw_image, captioner)
        log_captions(captions, file.name)
    
        references = images_labels[file.name]

        # concatenate all reference and caption attention images
        for caption in captions:
            attention_img = visualize_attention(raw_image, text_matcher, caption)

            # make caption visible in image
            Path(output_path / 'images' / file.stem).mkdir(parents=True, exist_ok=True)
            img_name = '_'.join(caption.split(' ')) + '_attention.png'
            pltimg.imsave(output_path / 'images' / file.stem / img_name, attention_img)
