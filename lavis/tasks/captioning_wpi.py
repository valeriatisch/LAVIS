from lavis.tasks.captioning import CaptionTask
from lavis.common.registry import registry
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from lavis.common.registry import registry
import socket
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import os
from word_forms.word_forms import get_word_forms
from nltk.stem import WordNetLemmatizer


@registry.register_task("captioning_wpi")
class CaptionWPITask(CaptionTask):
    def __init__(self, num_beams, max_len, min_len, evaluate, report_metric=True):
        super().__init__(num_beams, max_len, min_len, evaluate, report_metric)
        """
        Fetch 
        - wordnet and omw-1.4 for extracting synonyms
        - punkt for tokenizing
        - stopwords for removing stopwords

        Initialize lemmatizer, and stop words
        """
        nltk.download("wordnet")
        nltk.download("omw-1.4")
        nltk.download("punkt")
        nltk.download("stopwords")
        self.writer = None
        self.iteration = 0
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    def setup_writer(self):
        """
        Enable logging in tensorboard
        """

        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        log_dir_name = os.path.join(
            registry.get_path("output_dir"), "runs", current_time + socket.gethostname()
        )
        self.writer = SummaryWriter(log_dir=log_dir_name)

    def synonym_extractor(self, token: str):
        """
        Build set of synonyms from wordnet for a token
        Remove stopwords
        Apply lemmatization
        """
        synonyms = set()

        for syn in wordnet.synsets(token):
            synonyms.update(set(syn.lemma_names()))

        synonyms -= set(self.stop_words)
        synonyms = [self.lemmatizer.lemmatize(syn) for syn in synonyms]

        synonyms.append(token)
        return synonyms

    def related_words(self, words: list):
        """
        Build list with all possible words forms for words
        See get_word_forms (https://github.com/gutfeeling/word_forms)
        """
        related_words = list()
        for word in words:
            for word_family_values in get_word_forms(word).values():
                related_words.extend(list(word_family_values))
        return related_words

    def join_synonyms(self, tag: str):
        """
        Build synonym list for included words in a tag
        Synonym list consists of wordnet synonyms and the synonyms' related word forms
        """
        words = word_tokenize(tag)
        synonyms = set()
        for word in words:
            if any(char.isalpha() for char in word):
                synonyms.update(self.synonym_extractor(word))
                synonyms.update(self.related_words(synonyms))

        return synonyms

    def tags_synonyms(self, tags):
        """
        Build dictionary with tags as keys and corresponding synonyms as values
        """
        tag_synonyms_mapping = {}
        for tag in tags:
            if tag:
                tag_synonyms_mapping[tag] = self.join_synonyms(tag)
        return tag_synonyms_mapping

    def wpi_caption_eval(self, reference, caption: str):
        """
        wpi_caption_eval calculates similarity score between reference tags and caption
        - Tokenize caption, remove stop words and lemmatize the remaining words
        - Extract synonms for each tag and group them with corresponding tag
        - Check for each tag group whether a word of the group is in the caption
        - Calculate precision: matched tag groups / all tags
        """
        try:
            words = [
                token
                for token in word_tokenize(caption)
                if token not in self.stop_words
            ]
            words = [self.lemmatizer.lemmatize(token) for token in words]

        except Exception as e:
            return None
        tag_synonyms_mapping = self.tags_synonyms(reference)
        correct_count = 0
        tag_num = len(tag_synonyms_mapping.keys())
        if tag_num == 0:
            return None
        for tag, synonyms in tag_synonyms_mapping.items():
            if synonyms.intersection(words):
                correct_count += 1

        return correct_count / tag_num

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
        """
        Provides implementation for score calcultation defined in the base_task
        - Calculate precision for each caption in relation to reference tags
        - Calculate average precision
        """
        if self.writer == None:
            self.setup_writer()
        acc_accuracy = 0.0
        non_empty_refrences = 0

        for caption in val_result:
            single_acc = self.wpi_caption_eval(
                caption=caption["caption"], reference=caption["reference"]
            )
            if single_acc is not None:
                non_empty_refrences += 1
                acc_accuracy += self.wpi_caption_eval(
                    caption=caption["caption"], reference=caption["reference"]
                )
        score = acc_accuracy / non_empty_refrences

        self.writer.add_scalar("Synonym Accuracy", score, self.iteration)
        self.writer.flush()

        return {"synonym accuracy": score}
