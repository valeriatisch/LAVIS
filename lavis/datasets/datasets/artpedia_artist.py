"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import os
import json
from collections import OrderedDict

from lavis.datasets.datasets.coco_caption_datasets import COCOCapEvalDataset
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


class ArtistEvalDataset(COCOCapEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test

        Load annotations from annotation file
        Build artist set, that includes all artists in the annotations
        Format reference artist list
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        artists = set()
        for ann in self.annotation:
            ann['artist'] = [name.lower() for name in ann['artist']]
            artists.update(ann['artist'])
        self.artists = list(artists)

        for ann in self.annotation:
            ann['artist'] = [artist if artist in ann['artist'] else '' for artist in self.artists]


    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        print(ann['image_id'], image_path)

        image = self.vis_processor(image)
        return {
            "image": image,
            "image_id": ann["image_id"],
            "artists": ann["artist"],
        }
