"""
Implements evaluation for generated captions: 
Calculate a painting's scores for mentioned artists in relation to the ground truth artists
"""
import logging
from lavis.tasks.captioning import CaptionTask
from lavis.common.registry import registry
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from lavis.common.registry import registry
import socket
from datetime import datetime
import os
from sklearn.metrics import classification_report
from operator import itemgetter
import json


@registry.register_task("captioning_artist")
class CaptionArtistTask(CaptionTask):
    
    def __init__(self, num_beams, max_len, min_len, evaluate, report_metric=True):
        super().__init__(num_beams, max_len, min_len, evaluate, report_metric
        )
        self.writer = None
        self.iteration = 0

    # build datasets with settings from the configuration and retrieve all artists from the dataset
    def build_datasets(self, cfg):
        """
        Provides adjusted implementation defined in the base_task
        - Build pytorch dataset for the artist evaluation
        - Extract all artists part of the dataset (necessary for score eval)
        """
        datasets = dict()
        
        datasets_config = cfg.datasets_cfg

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            dataset_config = datasets_config[name]

            builder = registry.get_builder_class(name)(dataset_config)
            dataset = builder.build_datasets()
            
            datasets[name] = dataset
            self.artists = dataset['test'].artists

            logging.info(f'The dataset consists of {len(self.artists)} artists')


        return datasets

    def setup_writer(self):
        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        log_dir_name =  os.path.join(registry.get_path("output_dir"),
                "runs", current_time + socket.gethostname()
            ) 
        self.writer = SummaryWriter(log_dir=log_dir_name)

    def valid_step(self, model, samples):
        """
        Provides implementation defined in the base_task
        - Infer caption for each sample in the validation dataset
        - Map each caption to the artist reference
        """
        results = []

        captions = model.generate(
            samples,
            use_nucleus_sampling=True,
            max_length=self.max_len,
            min_length=self.min_len,
        )

        references = samples["artists"]
        
        for idx,caption in enumerate(captions):
            reference = list(map(itemgetter(idx), references))
            results.append({"caption": caption, "reference": reference})
        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        """
        Provides implementation for score calcultation defined in the base_task
        - Extract mentioned artists for each caption
        - Align included artists in the caption and reference artists format
        - Receive f1-score, precision and recall for each artist and
          the corresponeding average for all artists
        """
        if(self.writer == None):
            self.setup_writer()
        captions = [self.included_artists(caption = sample['caption']) for sample in val_result]
        references = [sample['reference'] for sample in val_result]
        caption_vectors = flatten(captions)       
        reference_vectors = flatten(references)
        non_empty_captions, non_empty_references = filter_empty_artists(reference_vectors=reference_vectors, caption_vectors=caption_vectors)

        # classification_report calculates precision, recall, f1score (totally and for each artist)
        metrics_report = classification_report(y_true = non_empty_references, y_pred = non_empty_captions, output_dict = True)

        result = metrics_report['macro avg']
        result['accuracy'] = metrics_report['accuracy']

        self.write_results(metrics_report, result)

        return result

    def write_results(self, metrics_report: dict, result: dict):
        """
        Results are written to the output runs folder specified in the config
        """
        with open (f'{self.writer.log_dir}/accuracy_report.json', 'w') as file:
            json.dump(metrics_report, file, indent = 4)

        for metric, score in result.items():
            self.writer.add_scalar(metric, score, self.iteration)
        
        logging.info(json.dumps(result, indent=4))
    
    # if the string includes an artist the artist's name is appended to the list, an empty string is appended
    def included_artists(self, caption: str):
        """
        Build list, with the included artists in a caption 
        - All artists part of the datasets are considered for included artist
        """
        caption = caption.lower()
        caption_vector = [ artist if artist in caption else '' for artist in self.artists]
        return caption_vector


def flatten(l: list):
    return [item for sublist in l for item in sublist]


def filter_empty_artists(reference_vectors: list, caption_vectors: list):
    """
    remove entries in the reference and caption list,
    where both caption and reference are empty
    """
    non_empty_references = []
    non_empty_captions = []
    for reference, caption in zip(reference_vectors, caption_vectors):
        if reference or caption:
            non_empty_references.append(reference)
            non_empty_captions.append(caption)
    return (non_empty_captions, non_empty_references)
