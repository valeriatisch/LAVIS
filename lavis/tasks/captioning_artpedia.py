from lavis.tasks.captioning import CaptionTask
from lavis.common.registry import registry
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from lavis.common.registry import registry
import socket
from datetime import datetime
import os

@registry.register_task("captioning_artpedia")
class CaptionArtpediaTask(CaptionTask):
    
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

    def coco_caption_eval(self, coco_gt_root, results_file, split):
        if(self.writer == None):
            self.setup_writer()

        results = super().coco_caption_eval(coco_gt_root, results_file, split)
        
        for metric, score in results.eval.items():
            self.writer.add_scalar(metric, score, self.iteration)
        self.writer.flush()

        return results

    def train_step(self, model, samples):
        if(self.writer == None):
            self.setup_writer()
        loss = super().train_step(model, samples)
        self.writer.add_scalar("train/loss", loss, self.iteration)
        self.writer.add_scalar("train/loss_item", loss.item(), self.iteration)
        self.iteration += 1
        self.writer.flush()
        return loss