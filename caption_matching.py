import os
from pathlib import Path

import json
import torch
from PIL import Image
from dotenv import load_dotenv

from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor

load_dotenv('.variables')

IMG_PATH = Path('/hpi/fs00/share/fg-naumann/seminar-ws22-tagging-captioning-art/artpedia-data/images/')
INPUT_FILE = Path('../artpedia/artpedia_res.json')
OUTPUT_FILE = Path('../artpedia/artpedia_scored.json')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, vis_processors, text_processors = load_model_and_preprocess("blip_image_text_matching",
                                                                   "large",
                                                                   device=device,
                                                                   is_eval=True)

with open(INPUT_FILE, "r") as json_file:
    annotations = json.load(json_file)

scored_annotations = {}
with open(os.getenv('SCORED_JSON_ANNOTATIONS'), 'w') as output_file:
    for i, entry in annotations.items():
        if entry['got_img'] == 'yes':
            file_name = entry['img_url'].split('/')[-1].split('.')[0] + '.png'
            raw_image = Image.open(IMG_PATH / file_name)  # .convert("RGB")
            img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
            captions = entry['visual_sentences']
            matching_scores = []
            cosine_similarities = []
            for caption in captions:
                txt = text_processors["eval"](caption)
                itm_output = model({"image": img, "text_input": txt}, match_head="itm")
                itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
                matching_scores.append(f'{itm_scores[:, 1].item():.3%}')
                itc_score = model({"image": img, "text_input": txt}, match_head='itc')
                cosine_similarities.append('%.4f'%itc_score)
            o_entry = entry
            o_entry['matching_scores'] = matching_scores
            o_entry['cosine_similarities'] = cosine_similarities
            # print(json.dumps(o_entry))
            scored_annotations[i] = entry
            scored_annotations[i]['matching_scores'] = matching_scores
            scored_annotations[i]['cosine_similarities'] = cosine_similarities

with open(OUTPUT_FILE, 'w') as f:
    json.dump(scored_annotations, f)
