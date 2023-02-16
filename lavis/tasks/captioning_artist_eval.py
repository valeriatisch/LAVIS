from lavis.tasks.captioning import CaptionTask
from lavis.common.registry import registry
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from lavis.common.registry import registry
import socket
from datetime import datetime
import os
from sklearn.metrics import classification_report


@registry.register_task("captioning_artist")
class CaptionArtistTask(CaptionTask):
    
    def __init__(self, num_beams, max_len, min_len, evaluate, report_metric=True):
        super().__init__(num_beams, max_len, min_len, evaluate, report_metric
        )
        self.writer = None
        self.iteration = 0

    def setup_writer(self):
        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        log_dir_name =  os.path.join(registry.get_path("output_dir"),
                "runs", current_time + socket.gethostname()
            ) 
        self.writer = SummaryWriter(log_dir=log_dir_name)

    def valid_step(self, model, samples):
        results = []

        captions = model.generate(
            samples,
            use_nucleus_sampling=True,
            max_length=self.max_len,
            min_length=self.min_len,
        )
        references = samples["artists"]
        all_artists = samples["all"]
        print('loaded bevore zip')
        for caption, reference, artist_set in zip(captions, references, all_artists):
            ref_captions = []
            results.append({"caption": caption, "reference": reference, "all": artist_set})
        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        captions = [sample['caption'] for sample in val_result]
        references = [sample['reference'] for sample in val_result]
        print(references[0] , '\n now the caption' ,captions[0])



        artists = list(val_result[0]['all'])
        print('artists', artists)
        caption_vectors= []
        for caption in captions:
            caption_vectors.append(self.format_caption(caption, artists= artists)) 
        reference_vectors = []
        for reference in references:
            reference_vectors.append([artist if artist in reference else '' for artist in artists])
        caption_vectors = self.flatten(caption_vectors)       
        reference_vectors = self.flatten(reference_vectors)
        print('caption_vectors: ', caption_vectors)
        print('reference_vectors: ', reference_vectors)

        metrics_report = classification_report(y_true = reference_vectors, y_pred = caption_vectors, output_dict = True)

        with open (f'{self.log_dir_name}/accuracy_report.json', 'w') as file:
            print(f'{self.log_dir_name}accuracy_report.json')
            json.dump(metrics_report, file, indent = 4)

        macro_avg = report['macro avg']
        result = report['accuracy']

        for metric, score in macro_avg.items():
            result[metric] = score
            self.writer.add_scalar(metric, score, self.iteration)
        self.writer.add_scalar('accuracy', accuracy)

        return result

    @staticmethod
    def flatten(l: list):
        return [item for sublist in l for item in sublist]
    

    def format_caption(caption: str, artists: list):
        caption = caption.lower()
        caption_vector = [ artist if artist in caption else '' for artist in artists]
        return caption_vector