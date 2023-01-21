import torch
from PIL import Image

from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor

# todos:
# model speichern
# artpedia dataset klasse nutzen
# alles filtern

img_path = '/dhc/home/smilla.fox/captioning-art-photographs-blip/img.jpg'

raw_image = Image.open(img_path).convert("RGB")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

caption = 'It is also stylistically earlier to that work, being painted without pseudo-perspective, and having the angels around the Virgin simply placed one above the other, rather than being spatially arranged.'
caption = 'There is a woman.'
model, vis_processors, text_processors = load_model_and_preprocess("blip_image_text_matching", "large", device=device, is_eval=True)

img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
txt = text_processors["eval"](caption)

itm_output = model({"image": img, "text_input": txt}, match_head="itm")
itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
print(f'The image and text are matched with a probability of {itm_scores[:, 1].item():.3%}')

itc_score = model({"image": img, "text_input": txt}, match_head='itc')

print('The image feature and text feature has a cosine similarity of %.4f'%itc_score)