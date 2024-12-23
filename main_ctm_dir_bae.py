from modules.data_reader import DataReader
from modules.data_saver import DataSaver
from modules.text_prep import TextPreparation
from modules.text_embeddings import TextEmbeddingGenerator
from modules.bow_embeddings import generate_bow
from modules.evaluation import Evaluation
from modules.ctm_dataset import CTMDataset
from modules.ctm_base import CTM
import time
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from modules.dir_vae_base_training import DIR_VAE

CONSTANTS = {
    "dataset":"scientific_papers",
    "embedding_method":"Pre-trained embeddings",
    "model":"GMM",
    "topic_method":"tf"
}


def run_experiment():
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    seed_everything(777)
    print("Reading Data")
    dr = DataReader()
    text_data = dr.obtain_text_data()
    print(text_data.shape)
    
    print("Preparing Text")
    tp = TextPreparation(text_data.en)
    prepped_text = tp.prepare_text()


    start = time.time()
    print("Calculating Embeddings")
    qt = TopicModelDataPreparation("paraphrase-multilingual-MiniLM-L12-v2")
    training_dataset = qt.fit(text_for_contextual=text_data.en, text_for_bow=prepped_text)
    model =  DIR_VAE(len(qt.vocab),384,20,num_epochs=20)
    # train
    early_stopping = EarlyStopping(monitor="val/loss",patience=5)
    model.fit(training_dataset)
    topics = model.predict(training_dataset)
    end = time.time()
    training_time = end-start

    print("Calculating Utilities")
    utils = Evaluation()
    utils.create_utility_objects(prepped_text)
    top_tokens = utils.get_top_topic_tokens(topics,method="freq")
    coherence = utils.get_coherence(top_tokens)
    diversity= utils.get_topic_diversity(top_tokens)
    other_stats = utils.get_dataset_stats(prepped_text)

    print(len(top_tokens))
    final_object = {"coherence":coherence,"top_tokens":top_tokens,
                    "diversity":diversity,**CONSTANTS,**other_stats
                    ,"vocab_size":tp.vocab_size,"embedding_model":"BERT","training_time":training_time}

    print("Saving...")
    data_saver = DataSaver()
    data_saver.save_object(final_object,"results_Dirichlet_test.json")
    print("Done")


if __name__ == "__main__":
    run_experiment()




