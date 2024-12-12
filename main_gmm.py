from modules.data_reader import DataReader
from modules.data_saver import DataSaver
from modules.text_prep import TextPreparation
from modules.text_embeddings import TextEmbeddingGenerator
from modules.topic_model import TopicModel
from modules.evaluation import Evaluation
import time

CONSTANTS = {
    "dataset":"scientific_papers",
    "embedding_method":"Pre-trained embeddings",
    "model":"GMM",
    "topic_method":"tf"
}


def run_experiment():
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

    print("Generating Topics")
    model = TopicModel(model_name="GMM")
    topics = model.get_topics(embeddings,save=True)

    del embeddings


    print("Calculating Utilities")
    print(topics)
    utils = Evaluation()
    utils.create_utility_objects(prepped_text)
    top_tokens = utils.get_top_topic_tokens(topics,method="freq")
    coherence = utils.get_coherence(top_tokens)


    diversity= utils.get_topic_diversity(top_tokens)
    
    other_stats = utils.get_dataset_stats(prepped_text)


    final_object = {"coherence":coherence,"top_tokens":top_tokens,
                    "diversity":diversity,**CONSTANTS,**other_stats,"vocab_size":tp.vocab_size,"embedding_model":embedding_model}

    print("Saving...")
    data_saver = DataSaver()
    data_saver.save_object(final_object,"results_GMM.json")
    print("Done")


if __name__ == "__main__":
    run_experiment()




