from modules.data_reader import DataReader
from modules.data_saver import DataSaver
from modules.text_prep import TextPreparation
from modules.text_embeddings import TextEmbeddingGenerator
from modules.bow_embeddings import generate_normalized_bow
from modules.evaluation import Evaluation
from modules.etmd import DVAE
#from modules.gaussian_vae import DVAE
#from modules.dir_vae import DVAE
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
    teg = TextEmbeddingGenerator(prepped_text)#,model="paraphrase-multilingual-mpnet-base-v2")
    embeddings = teg.calculate_embeddings()
    embedding_model = teg.model
    teg.unload_transformer()
    del teg
    print(embeddings)
    print(embeddings.std())
    print("Generating Topics")
    embeddings = (embeddings-np.min(embeddings))/(np.max(embeddings)-np.min(embeddings))
    #embeddings = (embeddings / embeddings.sum(1,keepdims=True))
    print(embeddings)
    big_train,test = train_test_split(embeddings,test_size=0.2,random_state=777)
    train,val = train_test_split(big_train,test_size=0.1,random_state=777)
    train_emb = torch.from_numpy(train).float()
    test_emb = torch.from_numpy(test).float()
    val_emb = torch.from_numpy(val).float()
    #mean, std = train_emb.mean(dim=0).to('cuda'), train_emb.std(dim=0)
    #train_emb_normalized = ((train_emb - mean) / std)
    #val_emb_normalized =  ((val_emb - mean) / std)
    train_dataloader = DataLoader(train_emb,batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_emb,batch_size=64, shuffle=True)
    model = DVAE(embedding_size=embeddings.shape[1], topic_size=20)
    # train
    early_stopping = EarlyStopping(monitor="val/loss")
    trainer = Trainer(max_epochs=200,
                      accelerator="auto", devices="auto", strategy="auto",callbacks=[early_stopping])
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    print(model(train_emb))
    topics = model.predict(train_emb)
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
                    ,"vocab_size":tp.vocab_size,"embedding_model":embedding_model,"training_time":training_time}

    print("Saving...")
    data_saver = DataSaver()
    data_saver.save_object(final_object,"results_DVAE_test.json")
    print("Done")


if __name__ == "__main__":
    run_experiment()




