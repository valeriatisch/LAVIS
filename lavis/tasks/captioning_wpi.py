from lavis.tasks.captioning import CaptionTask
from lavis.common.registry import registry
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from lavis.common.registry import registry
import socket
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from functools import reduce
import os
from word_forms.word_forms import get_word_forms
from wordhoard import Synonyms
from nltk.stem import WordNetLemmatizer


@registry.register_task("captioning_wpi")
class CaptionWPITask(CaptionTask):
    
    def __init__(self, num_beams, max_len, min_len, evaluate, report_metric=True):
        super().__init__(num_beams, max_len, min_len, evaluate, report_metric
        )
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        nltk.download('punkt')
        nltk.download('stopwords')
        self.writer = None
        self.iteration = 0
        self.ps = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def setup_writer(self):
        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        log_dir_name =  os.path.join(registry.get_path("output_dir"),
                "runs", current_time + socket.gethostname()
            ) 
        self.writer = SummaryWriter(log_dir=log_dir_name)

    def synonym_extractor(self, token: str):
        synonyms = [token]

        for syn in wordnet.synsets(token):
            for i in syn.lemma_names():
                new_synonyms = i.split('_')
                for synonym in new_synonyms:
                    if synonym not in self.stop_words:
                        synonyms.append(self.lemmatizer.lemmatize(synonym))

        return synonyms


    def synonym_extraction(self, token: str):
      synonyms = [token]
      syn_finder = Synonyms(token)
      syn_finder_words = syn_finder.find_synonyms()
      synonyms.extend(syn_finder_words)

      return synonyms


    def join_synonyms(self, tag: str):
        words = word_tokenize(tag)
        synonyms = set()
        for word in words:
            if any(char.isalpha() for char in word):
              word_form = self.synonym_extractor(word)
              for word_family_values in get_word_forms(word).values():
                word_form.extend(list(word_family_values))
              synonyms.update(word_form)
        
        return synonyms


    def tags_synonyms(self, tags):
        tag_synonyms_mapping = {}
        for tag in tags:
            if tag:
                tag_synonyms_mapping[tag] = self.join_synonyms(tag)        
        return tag_synonyms_mapping
    


    def wpi_caption_eval(self, reference, caption:str):
        try:
            words= [self.lemmatizer.lemmatize(toke) for toke in word_tokenize(caption) if toke not in self.stop_words]

        except Exception as e:
            return None
        tag_synonyms_mapping = self.tags_synonyms(reference)
        correct_count = 0
        sum = len(tag_synonyms_mapping.keys())
        if sum == 0:
            return None
        for tag, synonyms in tag_synonyms_mapping.items():
            if synonyms.intersection(words):
                correct_count += 1
            

        return correct_count / sum
    
    def valid_step(self, model, samples):
        results = []

        captions = model.generate(
            samples,
            use_nucleus_sampling=True,
            max_length=self.max_len,
            min_length=self.min_len,
        )
        references = samples["caption"]
        for caption, reference in zip(captions, references):
            ref_captions = []
            for tag in reference:
                if tag:
                    ref_captions.append(tag)
            results.append({"caption": caption, "reference": list(set(ref_captions))})
        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        if(self.writer == None):
            self.setup_writer()
        acc_accuracy = 0.0
        non_empty_refrences = 0

        for caption in val_result:
            single_acc = self.wpi_caption_eval(caption = caption["caption"], reference = caption["reference"])
            if single_acc is not None:
                non_empty_refrences += 1
                acc_accuracy += self.wpi_caption_eval(caption = caption["caption"], reference = caption["reference"])
        score = acc_accuracy / non_empty_refrences

        self.writer.add_scalar('Synonym Accuracy', score, self.iteration)
        self.writer.flush()
        
        return {'synonym accuracy': score}
