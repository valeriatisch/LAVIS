import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class WpiDataset(Dataset):
    def __init__(
            self,
            vis_processor=None,
            text_processor=None,
            vis_root=None,
            ann_paths=[],
            label_name='Topics',
            remove_documents=True):
        """
        Load annotations - Each annotation consists of:
        - otional: genres (e.g. ["notes (documents)"]), topics, places
        - required: title, image path
        - Label group for dataset reference is set with param label_name
        """
        annotations_file = ann_paths[0]
        self.anno_df = pd.read_json(annotations_file)
        self.img_dir = vis_root
        self.label_name = label_name
        self.labels_array = self.labels()
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        if remove_documents:
            self.remove_documents()
        if label_name == 'Genres':
            self.label_idx = 0
        elif label_name == 'Topics':
            self.label_idx = 5
        elif label_name == 'Places':
            self.label_idx = 6
        else:
            raise ValueError(
                'Must supply a valid label_name. Options are: Genres, Topics or Places')

    def labels(self):
        occurrences = dict()
        for _, labels in self.anno_df[self.label_name].dropna().items():
            for label in labels:
                occurrences[label] = occurrences[label] + \
                    1 if occurrences.get(label) != None else 1

        # remove samples with few occurences
        filtered_occ = {k: v for k, v in occurrences.items() if v >= 30}
        return list(filtered_occ.keys())

    def __len__(self):
        return len(self.anno_df) 
    
    def remove_documents(self):
        document_genres = ['notes (documents)', 'newspapers', 'books', 'newspaper columns', 'letters (correspondence)', 'autographs (manuscripts)', 'manuscripts (documents)', 'reports', 'sales catalogs', 'albums (books)', 'transcriptions (documents)', 'invoices', 'certificates', 'clippings (information artifacts)', 'receipts (financial records)', 'typescripts', 'publications (documents)', 'excerpts']
        removed_indices = []
        for idx in range(len(self.anno_df)):
            genres = self.anno_df.iloc[idx, 0]
            
            # handle missing labels
            if genres != genres:
                genres = []

            for genre in document_genres:
                if genre in genres:
                    removed_indices.append(idx)
                    break
        self.anno_df.drop(removed_indices, inplace=True)


    def __getitem__(self, idx):
        image_path = os.path.join(self.img_dir, self.anno_df.iloc[idx, 3])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        labels = self.anno_df.iloc[idx, self.label_idx]

        # handle missing labels
        if labels != labels:
            labels = []

        return {
            "image": image,
            "caption": [label if label in labels else "" for label in self.labels_array],
        }
