"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.coco_caption_datasets import (
    COCOCapDataset,
    COCOCapEvalDataset,
    NoCapsEvalDataset
)
from lavis.datasets.datasets.artpedia_filtered_dataset import FilterdArtpediaEvalDataset, FilteredArtpediaDataset

from lavis.common.registry import registry
from lavis.datasets.datasets.video_caption_datasets import (
    VideoCaptionDataset,
    VideoCaptionEvalDataset,
)
from lavis.datasets.datasets.caption_wpi_datasets import (
    WpiDataset
)
from lavis.datasets.datasets.artpedia_artist import (
    ArtistEvalDataset
)

@registry.register_builder("artpedia")
class ArtpediaBuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOCapDataset
    eval_dataset_cls = COCOCapEvalDataset

    DATASET_CONFIG_DICT = {
        "default" : "configs/datasets/artpedia/defaults_cap.yaml"
    }

@registry.register_builder("artpedia_artist")
class ArtpediaBuilder(BaseDatasetBuilder):
    eval_dataset_cls = ArtistEvalDataset

    DATASET_CONFIG_DICT = {
        "default" : "configs/datasets/artpedia/artist.yaml"
    }

@registry.register_builder("artpedia_filtered")
class ArtpediaFilteredBuilder(BaseDatasetBuilder):
    train_dataset_cls = FilteredArtpediaDataset
    eval_dataset_cls = COCOCapEvalDataset

    DATASET_CONFIG_DICT = {
        "default" : "configs/datasets/artpedia/filtered.yaml"
    }

    def build_dataset_class(self, vis_processor, text_processor, ann_paths, vis_path, is_train):
        filtering_threshold = self.config.get("filtering_threshold")
        dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
        if is_train:
            return dataset_cls(
                    vis_processor=vis_processor,
                    text_processor=text_processor,
                    ann_paths=ann_paths,
                    vis_root=vis_path,
                    filtering_threshold=filtering_threshold
                )
        else:
            return super().build_dataset_class(vis_processor, text_processor, ann_paths, vis_path, is_train)


@registry.register_builder("artpedia_bw")
class ArtpediaBuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOCapDataset
    eval_dataset_cls = COCOCapEvalDataset

    DATASET_CONFIG_DICT = {
        "default" : "configs/datasets/artpedia/bw_cap.yaml"
    }

@registry.register_builder("coco_caption")
class COCOCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOCapDataset
    eval_dataset_cls = COCOCapEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_cap.yaml",
    }


@registry.register_builder("nocaps")
class COCOCapBuilder(BaseDatasetBuilder):
    eval_dataset_cls = NoCapsEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/nocaps/defaults.yaml",
    }


@registry.register_builder("msrvtt_caption")
class MSRVTTCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msrvtt/defaults_cap.yaml",
    }


@registry.register_builder("msvd_caption")
class MSVDCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msvd/defaults_cap.yaml",
    }


@registry.register_builder("vatex_caption")
class VATEXCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vatex/defaults_cap.yaml",
    }

@registry.register_builder("wpi_caption")
class WPICapBuilder(BaseDatasetBuilder):
    eval_dataset_cls = WpiDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/wpi/defaults_cap.yaml",
    }