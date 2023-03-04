import json
import torch
import argparse
from PIL import Image
from pathlib import Path

from lavis.models import load_model_and_preprocess

# Define the command line arguments
parser = argparse.ArgumentParser(
    description="Generate matching scores and cosine similarities for image-caption pairs"
)
parser.add_argument(
    "images_dir", type=str, help="path to the input directory containing the images"
)
parser.add_argument(
    "annotations_json", type=str, help="path to the json file with annotations"
)
parser.add_argument(
    "output_json",
    type=str,
    help="path to the output json file containing matching scores and cosine similarities",
)
args = parser.parse_args()

IMG_PATH = Path(args.images_dir)
JSON_PATH = Path(args.annotations_json)
OUTPUT_JSON = Path(args.output_json)

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

    with open(OUTPUT_JSON, "w") as f:
        json.dump(scored_annotations, f, indent=4)
