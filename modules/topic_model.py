from sklearn.mixture import GaussianMixture,BayesianGaussianMixture
from sklearn.cluster import KMeans
import numpy as np

class TopicModel(object):

    def __init__(self,n_topics=20,model_name="GMM"):
        self.n_topics = n_topics
        if model_name == "GMM":
            self.model = BayesianGaussianMixture(n_components=n_topics,random_state=777,weight_concentration_prior_type='dirichlet_process')
        if model_name == "KM":
            self.model = KMeans(n_clusters=n_topics)
        self.model_name=model_name


    def _fit_model(self,embeddings):
        self.model.fit(embeddings)

    
    def _select_topics_GMM(self,embeddings):
        probs = self.model.predict_proba(embeddings)
        topics = [np.where(p > 1e-10) for p in probs]
        return probs.tolist(),topics
    
    def _select_topics_KM(self,embeddings):
        topics = self.model.predict(embeddings).tolist()
        topics_lists = [([k],) for k in topics]
        probs = list()
        for i in range(len(embeddings)):
            probs.append([1 if j == topics[i] else 0 for j in range(self.n_topics)])
        return probs,topics_lists

    def _select_topics(self,embeddings,save=True):
        model_function_dict = {"GMM":self._select_topics_GMM,"KM":self._select_topics_KM}
        probs,topics = model_function_dict[self.model_name](embeddings)
        if save:
            self.topics = topics
            self.probs = probs
    

    def get_topics(self,embeddings,save=True):
        if save == True:
            self._fit_model(embeddings)
        self._select_topics(embeddings,save)
        return self.topics