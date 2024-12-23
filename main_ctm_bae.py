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
from modules.student_pred_model import Student_Pred

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
    indices = np.arange(0,prepped_text.shape[0])
    big_train,test = train_test_split(indices,test_size=0.05,random_state=777)
    train,val = train_test_split(big_train,test_size=0.1,random_state=777)
    train_data_contextual  = text_data.en.iloc[train].reset_index(drop=True)
    train_data_for_bow =prepped_text.iloc[train].reset_index(drop=True)
    val_data_contextual = text_data.en.iloc[val].reset_index(drop=True)
    val_data_for_bow =prepped_text.iloc[val].reset_index(drop=True)
    training_dataset = qt.fit(text_for_contextual=text_data.en, text_for_bow=prepped_text)
    validation_dataset = qt.transform(val_data_contextual,val_data_for_bow)
    model = CTM(len(qt.vocab),384,20,num_epochs=20)
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
    data_saver.save_object(final_object,"results_DVAE_test.json")
    teacher_output = model.get_doc_topic_distribution(training_dataset)
    teg = TextEmbeddingGenerator(prepped_text)
    student_embeddings = teg.bert_embeddings_from_list(text_data.en,'paraphrase-multilingual-MiniLM-L12-v2')
    student_model = Student_Pred(student_embeddings.shape[1],teacher_output.shape[1])
    student_dataset = TensorDataset(student_embeddings,teacher_output)
    student_dataloader = DataLoader(student_dataset,batch_size=64, shuffle=True)
    trainer = Trainer(max_epochs=20,
                      accelerator="auto", devices="auto", strategy="auto")#,callbacks=[early_stopping])
    trainer.fit(model=student_model, train_dataloaders=student_dataloader)



    print("Done")


if __name__ == "__main__":
    run_experiment()




