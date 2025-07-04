from modules.data_reader import DataReader
from modules.data_saver import DataSaver
from modules.text_prep import TextPreparation
from modules.evaluation import Evaluation
from modules.bertopic_model import TopicModel
from nltk.tokenize import word_tokenize
import time
from pytorch_lightning import  seed_everything
import pandas as pd

CONSTANTS = {
    "dataset":"news",
    "embedding_method":"CTM",
    "model":"CTM",
    "topic_method":"CTM",
    "language":"indonesian"
}


files = [
    {"teacher":"news_docs_id_2017","student":"news_docs_id_2017"},
    {"teacher":"news_docs_id_2018","student":"news_docs_id_2018"},

]

vocab_sizes = {
    "md":{"min_df":0.01,"max_df":0.5},
}


seeds = [7,42,77,420,777]

def run_experiment():
    print("Reading Data")
    for seed in seeds:
            for topic_size in [20]:
                seed_everything(seed)
                full_data = list()
                for file in files:
                    dr = DataReader(filename=f"{file['student']}.csv")
                    data_j = dr.obtain_text_data().dropna()
                    data_j["timestamp"] = file["student"].split("_")[-1] + "-01-01"
                    full_data.append(data_j)
                    text_data = pd.concat(full_data).reset_index(drop=True)
                
                print("Preparing Text")
                tp = TextPreparation(text_data.en,language="english",cv_params={"min_df":0.01,"max_df":0.5})
                prepped_text = tp.prepare_text()

                print("Generating Topics")
                start = time.time()
                try:
                    model = TopicModel(topic_size)
                except:
                    continue
                topics = model.get_topics(prepped_text.values.tolist())
                end = time.time()
                training_time = end-start


                print("Calculating Utilities")
                print(topics)
                topic_time = topics.topics_over_time(prepped_text.values.tolist(),text_data.timestamp.values.tolist(),nr_bins=len(files))
                print(topic_time)
                timestamps = (topic_time.Timestamp.dt.year+1).unique().tolist()
                for i,ts in enumerate(timestamps):
                    top_tokens = topic_time.loc[(topic_time.Timestamp.dt.year+1) == ts ].Words.values
                    top_tokens = [t.split(", ") for t in top_tokens]
                    utils = Evaluation(n_topics=topic_size)
                    utils.create_utility_objects(prepped_text)

                    coherence = utils.get_coherence(top_tokens)
                    diversity= utils.get_topic_diversity(top_tokens)
                    other_stats = utils.get_dataset_stats(prepped_text)


                    final_object = {"coherence":coherence,"top_tokens":top_tokens,
                                    "diversity":diversity,**CONSTANTS,**other_stats,"vocab_size":tp.vocab_size,"training_time":training_time}

                    print("Saving...")
                    data_saver = DataSaver()
                    file = files[i]["student"]
                    data_saver.save_object(final_object,f"results/results_BT_{file}_{seed}_{topic_size}.json")
                print("Done")
                raise ValueError


if __name__ == "__main__":
    run_experiment()




