from .coco_caption_datasets import CaptionDataset, CaptionEvalDataset


class FilteredArtpediaDataset(CaptionDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        filtered_annotations = []
        for annotation in self.annotation:
            if annotation["matching_score"] >= 0.8:
                filtered_annotations.append(annotation)
        self.annotation = filtered_annotations


class FilterdArtpediaEvalDataset(CaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        filtered_annotations = []
        for annotation in self.annotation:
            if annotation["matching_score"] >= 0.8:
                filtered_annotations.append(annotation)
        self.annotation = filtered_annotations
