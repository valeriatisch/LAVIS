"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
from collections import OrderedDict

from lavis.datasets.datasets.coco_caption_datasets import COCOCapEvalDataset
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


class CaptionEvalArtistDataset(COCOCapEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        artists = set()
        for ann in self.annotation:
            print(ann['artist'])
            ann['artist'] = [name.lower() for name in ann['artist']]
            artists.update(ann['artist'])
        self.artists = list(artists)

        for ann in self.annotation:
            ann['artist'] = self.format_reference(ann['artist'])
    

    def format_reference(self, reference: list):
        reference_vector = []
        for artist in self.artists:
            if artist in reference:
                reference_vector.append([artist, True])
            else:
                reference_vector.append([artist, False])
        return reference_vector


    def __getitem__(self, index):
        ann = self.annotation[index]
        print(ann['image'])


        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        print ('check', {
            "image": image,
            "image_id": ann["image_id"],
            "artists": self.format_reference(ann["artist"]),
        })
        return {
            "image": image,
            "image_id": ann["image_id"],
            "artists": ann["artist"],
            "all": self.artists
        }
