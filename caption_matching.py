import os
from pathlib import Path
import json
import torch
from PIL import Image
from dotenv import load_dotenv

from lavis.models import load_model_and_preprocess


load_dotenv(".variables")

IMG_PATH = Path(os.getenv("IMAGES_PATH"))
JSON_PATH = Path(os.getenv("JSON_ANNOTATIONS_PATH"))


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load classifier for image text matching
    model, vis_processors, text_processors = load_model_and_preprocess(
        "blip_image_text_matching", "large", device=device, is_eval=True
    )

    # Load ground-truth captions
    with open(JSON_PATH, "r") as json_file:
        annotations = json.load(json_file)

    scored_annotations = []

    for entry in annotations:
        file_name = entry["image"]
        raw_image = Image.open(IMG_PATH / file_name)
        caption = entry["caption"]

        # Process image and text
        processed_img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        processed_text = text_processors["eval"](caption)

        # Calculate matching score
        itm_output = model(
            {"image": processed_img, "text_input": processed_text}, match_head="itm"
        )
        itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
        matching_score = itm_scores[:, 1].item()

        # Calculate cosine similarity
        itc_score = model(
            {"image": processed_img, "text_input": processed_text}, match_head="itc"
        )
        cosine_similarity = float(itc_score.cpu().detach().numpy()[0][0])

        entry["matching_score"] = matching_score
        entry["cosine_similarity"] = cosine_similarity
        scored_annotations.append(entry)

    with open(os.getenv("SCORED_JSON_ANNOTATIONS"), "w") as f:
        json.dump(scored_annotations, f, indent=4)
