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
    "dataset":"news",
    "embedding_method":"Pre-trained embeddings",
    "model":"GMM",
    "topic_method":"tf",
    "language":"indonesian"
}

files = [
    "news_docs_id_2018",

]


betas = {
    "balanced":{"rl":1,"kl":1,"dl":1,"al":1},
    "kl_focused":{"rl":0.3,"kl":1,"dl":0.3,"al":0.3},
    "dl_focused":{"rl":0.3,"kl":0.3,"dl":1,"al":0.3},
    "al_focused": {"rl":0.3,"kl":0.3,"dl":0.3,"al":1},
    "rl_focused":{"rl":1,"kl":0.3,"dl":0.3,"al":0.3},
}

def run_experiment():
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    seed_everything(777)
    print("Reading Data")
    dr = DataReader(filename="news_docs_en_2015.csv")
    text_data = dr.obtain_text_data()
    print(text_data.shape)
    
    print("Preparing Text")
    tp = TextPreparation(text_data.en,cv_params={"min_df":0.01,"max_df":0.5})
    prepped_text = tp.prepare_text()

    start = time.time()
    print("Calculating Embeddings")
    qt = TopicModelDataPreparation("paraphrase-multilingual-MiniLM-L12-v2")
    indices = np.arange(0,prepped_text.shape[0])
    big_train,test = train_test_split(indices,test_size=0.1,random_state=777)
    train,val = train_test_split(big_train,test_size=0.1,random_state=777)
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
    model =  DIR_VAE(len(qt.vocab),384,20,num_epochs=50)
    # train
    early_stopping = EarlyStopping(monitor="val/loss",patience=5)
    model.fit(training_dataset,validation_dataset=val_dataset)

    end = time.time()
    training_time = end-start
    for file in files:
        for vocab_size,params in betas.items():
            save_file_name = f"TS_{file}_{vocab_size}"
            dr = DataReader(filename=f"{file}.csv")
            es_text_data = dr.obtain_text_data()
            print(es_text_data.shape)

            print("Preparing Text")
            es_tp = TextPreparation(es_text_data.en,language='english',cv_params={"min_df":0.01,"max_df":0.5})
            prepped_text_es = es_tp.prepare_text()
            print(prepped_text_es.shape)
            teg = TextEmbeddingGenerator(prepped_text_es)

            
            qt_sp = TopicModelDataPreparation("paraphrase-multilingual-MiniLM-L12-v2")
            indices_sp = np.arange(0,prepped_text_es.shape[0])
            big_train_sp,test_sp = train_test_split(indices_sp,test_size=0.1,random_state=777)
            train_sp,val_sp = train_test_split(big_train_sp,test_size=0.1,random_state=777)
            train_contextual_sp =  es_text_data.en.loc[train_sp].reset_index(drop=True)
            train_bow_prepped_sp = prepped_text_es.loc[train_sp].reset_index(drop=True)
            training_dataset_sp = qt_sp.fit(text_for_contextual=train_contextual_sp
                                    , text_for_bow=train_bow_prepped_sp)
            val_contextual_sp =  es_text_data.en.loc[val_sp].reset_index(drop=True)
            val_bow_prepped_sp = prepped_text_es.loc[val_sp].reset_index(drop=True)
            val_dataset_sp = qt_sp.transform(text_for_contextual=val_contextual_sp
                                    , text_for_bow=val_bow_prepped_sp)
            posterior = model.get_posterior(training_dataset_sp).mean(axis=0)


            test_contextual_sp =  es_text_data.en.loc[test_sp].reset_index(drop=True)
            test_bow_prepped_sp = prepped_text_es.loc[test_sp].reset_index(drop=True)
            test_dataset_sp = qt_sp.transform(text_for_contextual=test_contextual_sp
                                    , text_for_bow=test_bow_prepped_sp)
            student_model =  DIR_VAE(len(qt_sp.vocab),384,20,num_epochs=50,prior=posterior,
                                    training_texts=train_bow_prepped_sp,id2token=qt_sp.id2token,teacher=model.model,
                                    beta=params) 
            student_model.fit(training_dataset_sp,val_dataset_sp)
            topics = student_model.predict(test_dataset_sp)
            teacher_posterior = model.get_posterior(test_dataset_sp)
            student_posterior = student_model.get_posterior(test_dataset_sp)
            #topics = model.predict(training_dataset)
            print("Calculating Utilities")
            utils = Evaluation()
            utils.create_utility_objects(train_bow_prepped_sp)
            top_tokens = student_model.get_top_tokens(qt_sp.id2token)
            print(len(top_tokens))
            coherence = utils.get_coherence(top_tokens)
            diversity= utils.get_topic_diversity(top_tokens)
            other_stats = utils.get_dataset_stats(test_bow_prepped_sp)
            # Plot training KL Divergence over time.
            utils.plot_losses(student_model.training_losses,save_file_name)

            teacher_topics = model.predict(test_dataset)
            teacher_utils = Evaluation()
            teacher_utils.create_utility_objects(train_bow_prepped)
            teacher_top_tokens = model.get_top_tokens(qt.id2token)
            teacher_coherence = teacher_utils.get_coherence(teacher_top_tokens)
            teacher_diversity= teacher_utils.get_topic_diversity(teacher_top_tokens)
            cross_lingual_similarity = student_model._student_teacher_topic_similarity(model.model.encoder,student_model.model.encoder).detach().cpu().numpy().tolist()
            print(cross_lingual_similarity)
            cross_lingual_toptoken_similarity = utils.get_similarity_top_tokens(teacher_top_tokens,top_tokens)
            final_object = {"coherence":coherence,"teacher_coherence":teacher_coherence,"top_tokens":top_tokens,"teacher_top_tokens":teacher_top_tokens,
                            "diversity":diversity,"teacher_diversity":teacher_diversity,"cls":cross_lingual_similarity,
                            "cross_lingual_toptoken_similarity":cross_lingual_toptoken_similarity,**CONSTANTS,**other_stats
                            ,"vocab_size":tp.vocab_size,"embedding_model":"BERT","training_time":training_time}

            print("Saving...")
            data_saver = DataSaver()
            data_saver.save_object(final_object,f"results/{save_file_name}.json")
            print("Done")


if __name__ == "__main__":
    run_experiment()




