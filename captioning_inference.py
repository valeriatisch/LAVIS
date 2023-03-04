import requests
import torch
from PIL import Image

from lavis.models import load_model_and_preprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# loads BLIP caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
# this also loads the associated image processors
model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True,
                                                     device=device)
# preprocess the image
# vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)


img_url = "https://uploads6.wikiart.org/images/salvador-dali/the-persistence-of-memory-1931.jpg!Large.jpg"
img = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
image = vis_processors["eval"](img).unsqueeze(0).to(device)
# generate caption
print(model.generate({"image": image}, use_nucleus_sampling=True, num_captions=3))
