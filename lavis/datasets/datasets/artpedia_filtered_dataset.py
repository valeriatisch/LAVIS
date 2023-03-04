from .coco_caption_datasets import CaptionDataset, CaptionEvalDataset


class FilteredArtpediaDataset(CaptionDataset):
    def __init__(
        self,
        vis_processor,
        text_processor,
        image_directory,
        annotations_path,
        filtering_threshold,
    ):
        super().__init__(
            vis_processor, text_processor, image_directory, annotations_path
        )
        filtered_annotations = []
        for annotation in self.annotation:
            if annotation["matching_score"] >= filtering_threshold:
                filtered_annotations.append(annotation)
        self.annotation = filtered_annotations


class FilterdArtpediaEvalDataset(CaptionEvalDataset):
    def __init__(
        self,
        vis_processor,
        text_processor,
        image_directory,
        annotations_path,
        filtering_threshold,
    ):
        super().__init__(
            vis_processor, text_processor, image_directory, annotations_path
        )
        filtered_annotations = []
        for annotation in self.annotation:
            if annotation["matching_score"] >= filtering_threshold:
                filtered_annotations.append(annotation)
        self.annotation = filtered_annotations
