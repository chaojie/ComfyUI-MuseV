from typing import List, Tuple, Dict
import os
import time

from tqdm import tqdm
import torch
import numpy as np
from numpy import ndarray
from PIL import Image
from transformers import BertForSequenceClassification, BertTokenizer, CLIPProcessor, CLIPModel



class TextFeatureExtractor(object):
    def __init__(self, language_model_path: str, local_file: bool=True, device: str='cpu'):
        if device:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        language_model_path = "Taiyi-CLIP-Roberta-large-326M-Chinese" if local_file else "IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese"    
        self.text_tokenizer = BertTokenizer.from_pretrained(language_model_path, local_files_only=local_file)
        self.text_encoder = BertForSequenceClassification.from_pretrained(language_model_path, local_files_only=local_file).eval().to(self.device)
   
    def text(self, query_texts: List[str]) -> ndarray:
        text = self.text_tokenizer(query_texts, return_tensors='pt', padding=True, truncation=True, max_length=self.text_encoder.config.max_length)['input_ids']
        text = text.to(self.device)
        with torch.no_grad():
            text_features = self.text_encoder(text).logits
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            text_features = text_features.squeeze
        return text_features.detach().cpu().numpy()


class TaiyiFeatureExtractor(TextFeatureExtractor):
    def __init__(self, language_model_path: str="Taiyi-CLIP-Roberta-large-326M-Chinese", local_file: bool = True, device: str = 'cpu'):
        """_summary_

        Args:
            language_model_path (str, optional): Taiyi-CLIP-Roberta-large-326M-Chinese or IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese. Defaults to "Taiyi-CLIP-Roberta-large-326M-Chinese".
            local_file (bool, optional): _description_. Defaults to True.
            device (str, optional): _description_. Defaults to 'cpu'.
        """
        super().__init__(language_model_path, local_file, device)
        


