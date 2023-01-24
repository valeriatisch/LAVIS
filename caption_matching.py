import os
from pathlib import Path
 
import numpy as np
import json
import torch
from PIL import Image
from dotenv import load_dotenv

from lavis.models import load_model_and_preprocess

load_dotenv('.variables')

IMG_PATH = Path(os.getenv('IMAGES_PATH'))
JSON_PATH = Path(os.getenv('JSON_ANNOTATIONS_PATH'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, vis_processors, text_processors = load_model_and_preprocess("blip_image_text_matching",
                                                                   "large",
                                                                   device=device,
                                                                   is_eval=True)

with open(JSON_PATH, "r") as json_file:
    annotations = json.load(json_file)

scored_annotations = []
counter = 0
for entry in annotations:
    if counter % 100 == 0:
        print(counter)
    file_name = entry["image"]
    raw_image = Image.open(IMG_PATH / file_name) # .convert("RGB")
    img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    caption = entry['caption']
    txt = text_processors["eval"](caption)
    itm_output = model({"image": img, "text_input": txt}, match_head="itm")
    itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
    matching_score = itm_scores[:, 1].item()
    itc_score = model({"image": img, "text_input": txt}, match_head='itc')
    cosine_similarity = float(itc_score.cpu().detach().numpy()[0][0])
    entry['matching_score'] = matching_score
    entry['cosine_similarity'] = cosine_similarity
    scored_annotations.append(entry)
    counter += 1

with open(os.getenv('SCORED_JSON_ANNOTATIONS'), 'w') as f:
    json.dump(scored_annotations, f, indent=4)

# print(f'The image and text are matched with a probability of {itm_scores[:, 1].item():.3%}')
# print('The image feature and text feature has a cosine similarity of %.4f'%itc_score)
