from contextualized_topic_models.models.ctm import ZeroShotTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
import numpy as np

class TopicModel(object):

    def __init__(self,n_topics=20,bert_version="paraphrase-multilingual-MiniLM-L12-v2"):
        self.n_topics = n_topics
        self.bert_version =  bert_version
        self.model_name="CTM"


    def _fit_model(self,text_for_contextual,text_for_bow):
        qt = TopicModelDataPreparation(self.bert_version)
        training_dataset = qt.fit(text_for_contextual=text_for_contextual, text_for_bow=text_for_bow)
        ctm = ZeroShotTM(bow_size=len(qt.vocab), contextual_size=384, n_components=self.n_topics,num_data_loader_workers=1,hidden_sizes=(100, 100),num_epochs=20)
        ctm.fit(training_dataset)
        self.qt = qt
        self.model = ctm

    def _select_topics(self,text_for_contextual):
        testing_dataset = self.qt.transform(text_for_contextual=text_for_contextual)
        # n_sample how many times to sample the distribution (see the doc)
        topics= self.model.get_doc_topic_distribution(testing_dataset, n_samples=10)
        self.topics = [np.where(p > 0.1) if len(np.where(p > 0.1)[0]) > 0 else np.where(p>=0.01) for p in topics]



    def get_topics(self,text_for_contextual,text_for_bow=list(),save=True):
        if save == True:
            self._fit_model(text_for_contextual,text_for_bow)
        print("Predicting")
        self._select_topics(text_for_contextual)
        return self.topics