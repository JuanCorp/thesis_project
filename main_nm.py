from modules.data_reader import DataReader
from modules.data_saver import DataSaver
from modules.text_prep import TextPreparation
from modules.evaluation import Evaluation
from modules.neuromax_trainer import BasicTrainer
from modules.NeuroMax import NeuroMax
from nltk.tokenize import word_tokenize
import time
from pytorch_lightning import  seed_everything
import numpy as np
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from sklearn.model_selection import train_test_split
import torch



CONSTANTS = {
    "dataset":"news",
    "embedding_method":"CTM",
    "model":"CTM",
    "topic_method":"CTM",
    "language":"indonesian"
}


files = [
    "news_docs_id_2018",

]

vocab_sizes = {
    "sm":{"min_df":0.1,"max_df":0.5},
    "md":{"min_df":0.01,"max_df":0.5},
    "lg":{"min_df":0.001,"max_df":0.5}
}


seeds = [777]

def run_experiment():
    print("Reading Data")
    for seed in seeds:
        for file in files:
            for topic_size in [20,50]:
                    seed_everything(seed)
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
                    model =  NeuroMax(len(qt.vocab),topic_size)
                    device = (
                    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                )
                    model.to(device)

                    trainer = BasicTrainer(model,epochs=50)
                    # train
                    trainer.train(training_dataset)

                    end = time.time()
                    training_time = end-start
                    teacher_utils = Evaluation(n_topics=topic_size)
                    teacher_utils.create_utility_objects(train_bow_prepped)
                    teacher_top_tokens = trainer.export_top_words(qt.id2token)
                    teacher_coherence = teacher_utils.get_coherence(teacher_top_tokens)
                    teacher_diversity= teacher_utils.get_topic_diversity(teacher_top_tokens)
                    other_stats = teacher_utils.get_dataset_stats(train_bow_prepped)
                    final_object = {"coherence":teacher_coherence,"top_tokens":teacher_top_tokens,
                                    "diversity":teacher_diversity,**CONSTANTS,**other_stats
                                    ,"vocab_size":tp.vocab_size,"embedding_model":"BERT","training_time":training_time}

                    print("Saving...")
                    data_saver = DataSaver()
                    data_saver.save_object(final_object,f"results/results_NM_{file}_{seed}_{topic_size}.json")
                    print("Done")


if __name__ == "__main__":
    run_experiment()




