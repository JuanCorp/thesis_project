from modules.data_reader import DataReader
from modules.data_saver import DataSaver
from modules.text_prep import TextPreparation
from modules.text_embeddings import TextEmbeddingGenerator
from modules.bow_embeddings import generate_bow
from modules.evaluation import Evaluation
from modules.student_ctm_dataset import CTMDataset
from modules.ctm_base import CTM
import time
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from modules.dir_vae_base_training import DIR_VAE

CONSTANTS = {
    "dataset":"ted_talks",
    "embedding_method":"Pre-trained embeddings",
    "model":"GMM",
    "topic_method":"tf",
    "language":"urdu"
}

files = [
    "news_docs_ht_2018"

]

vocab_sizes = {
    "md":{"min_df":0.01,"max_df":0.5},

}

seeds = [777]


def run_experiment():
    for seed in seeds:
        seed_everything(seed)
        for topic_size in [20]:
            for file in files:
                    dr = DataReader(filename=f"{file}.csv")
                    text_data = dr.obtain_text_data()
                    print(text_data.shape)
                    
                    print("Preparing Text")
                    tp = TextPreparation(text_data.en,cv_params={"min_df":0.01,"max_df":0.5},language="english")
                    prepped_text = tp.prepare_text()
                    print(prepped_text.shape)
                    start = time.time()
                    print("Calculating Embeddings")
                    qt = TopicModelDataPreparation("paraphrase-multilingual-MiniLM-L12-v2")
                    indices = np.arange(0,prepped_text.shape[0])
                    big_train,test = train_test_split(indices,test_size=0.1,random_state=seed)
                    train,val = train_test_split(big_train,test_size=0.1,random_state=seed)
                    train_contextual =  text_data.en.loc[train].reset_index(drop=True)
                    train_bow_prepped = prepped_text.loc[train].reset_index(drop=True)
                    training_dataset = qt.fit(text_for_contextual=train_contextual
                                            , text_for_bow=train_bow_prepped)
                    val_contextual =  text_data.en.loc[val].reset_index(drop=True)
                    val_bow_prepped = prepped_text.loc[val].reset_index(drop=True)
                    val_dataset = qt.transform(text_for_contextual=val_contextual
                                            , text_for_bow=val_bow_prepped)
                    test_contextual =  text_data.en.loc[test].reset_index(drop=True)
                    test_bow_prepped = prepped_text.loc[test].reset_index(drop=True)
                    test_dataset = qt.transform(text_for_contextual=test_contextual
                                            , text_for_bow=test_bow_prepped)
                    model =  DIR_VAE(len(qt.vocab),384,topic_size,num_epochs=50)
                    # train
                    early_stopping = EarlyStopping(monitor="val/loss",patience=5)
                    model.fit(training_dataset,validation_dataset=val_dataset)

                    end = time.time()
                    training_time = end-start

                    teacher_topics = model.predict(test_dataset)
                    teacher_utils = Evaluation(n_topics=topic_size)
                    teacher_utils.create_utility_objects(train_bow_prepped)
                    teacher_top_tokens = model.get_top_tokens(qt.id2token)
                    coherences = list()
                    teacher_coherence = teacher_utils.get_coherence(teacher_top_tokens)
                    for tt in teacher_top_tokens:
                         coherences.append(teacher_utils.get_coherence([tt]))
                    teacher_diversity= teacher_utils.get_topic_diversity(teacher_top_tokens)
                    other_stats = teacher_utils.get_dataset_stats(train_bow_prepped)
                    final_object = {"teacher_coherence":teacher_coherence,"teacher_top_tokens":teacher_top_tokens,
                                    "teacher_diversity":teacher_diversity,**CONSTANTS,**other_stats
                                    ,"vocab_size":tp.vocab_size,"embedding_model":"BERT","training_time":training_time,
                                    "coherences":coherences}

                    print("Saving...")
                    data_saver = DataSaver()
                    data_saver.save_object(final_object,f"results/results_DIR_temp_{file}_{seed}_{topic_size}.json")
                    print("Done")


if __name__ == "__main__":
    run_experiment()




