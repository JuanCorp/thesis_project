from modules.data_reader import DataReader
from modules.data_saver import DataSaver
from modules.text_prep import TextPreparation
from modules.evaluation import Evaluation
from modules.ctm_model import TopicModel
from nltk.tokenize import word_tokenize
import time


CONSTANTS = {
    "dataset":"UN_100w",
    "embedding_method":"CTM",
    "model":"CTM",
    "topic_method":"CTM"
}

def run_experiment():
    print("Reading Data")
    dr = DataReader()
    text_data = dr.obtain_text_data()
    print(text_data.shape)
    
    print("Preparing Text")
    tp = TextPreparation(text_data.en)
    prepped_text = tp.prepare_text()

    print("Generating Topics")
    start = time.time()
    model = TopicModel()
    topics = model.get_topics(text_data.en.values.tolist(),prepped_text.values.tolist(),save=True)
    end = time.time()
    training_time = end-start


    print("Calculating Utilities")
    print(topics)
    utils = Evaluation()
    utils.create_utility_objects(prepped_text)
    top_tokens = utils.get_top_topic_tokens(topics)
    coherence = utils.get_coherence(top_tokens)



    diversity= utils.get_topic_diversity(top_tokens)
    


    other_stats = utils.get_dataset_stats(prepped_text)


    final_object = {"coherence":coherence,"top_tokens":top_tokens,
                    "diversity":diversity,**CONSTANTS,**other_stats,"vocab_size":tp.vocab_size,"training_time":training_time}

    print("Saving...")
    data_saver = DataSaver()
    data_saver.save_object(final_object,"results_CTM.json")
    print("Done")


if __name__ == "__main__":
    run_experiment()



