from bertopic import BERTopic
import numpy as np
from sklearn.cluster import KMeans

class TopicModel(object):

    def __init__(self,n_topics=20):
        self.n_topics = n_topics
        self.model_name="Bertopic"





    def get_topics(self,text_for_bow=list()):
        kmeans = KMeans(n_clusters=self.n_topics)
        self.model = BERTopic(hdbscan_model=kmeans,language="multilingual",)
        self.model.fit(text_for_bow)
        return self.model
    
    def get_toptokens(self):
        topics =  self.model.get_topics()
        topic_words = list()
        for t,words in topics.items():
            if t!=-1:
                t_words = [w[0] for w in words]
                topic_words.append(t_words)
        return topic_words