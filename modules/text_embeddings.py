from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import numpy as np
import nltk

class TextEmbeddingGenerator(object):

    def __init__(self,model='paraphrase-multilingual-MiniLM-L12-v2'):
        self.model = model
        nltk.download('punkt')

    def _load_transformer(self):
        self.transformer = SentenceTransformer(self.model,device='cuda:0')

    def unload_transformer(self):
        del self.transformer


    def _generate_text_embeddings(self,texts):
        text_embeddings = list()
        for text in texts:
            tokenized_sentences = sent_tokenize(text)
            sentence_embeddings = self.transformer.encode(tokenized_sentences)
            averaged_sentences = sentence_embeddings.mean(axis=0)
            text_embeddings.append(averaged_sentences)
        
        return np.array(text_embeddings)
    

    def calculate_embeddings(self,texts):
        self._load_transformer()
        embeddings = self._generate_text_embeddings(texts)
        return embeddings
    


    def bert_embeddings_from_list(
        self,texts, sbert_model_to_load, batch_size=200, max_seq_length=None
    ):

        model = SentenceTransformer(sbert_model_to_load)

        if max_seq_length is not None:
            model.max_seq_length = max_seq_length


        return np.array(model.encode(texts, show_progress_bar=True, batch_size=batch_size))
        

        
        