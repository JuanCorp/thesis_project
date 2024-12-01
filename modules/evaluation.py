import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
import numpy as np
from collections import Counter
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class Evaluation(object):

    def __init__(self,coherence_mode="c_npmi",n_topics=20,topk=10):
        self.coherence_mode = coherence_mode
        self.n_topics = n_topics
        self.topk = topk


    def create_utility_objects(self,data):
        self.tokenized_word_sentences = [word_tokenize(s) for s in data.values]
        self.data = data
        self.id2word = corpora.Dictionary(self.tokenized_word_sentences)

    

    def get_top_topic_tokens(self,topics,method="freq"):
        topic_top_n = list()
        for topic in range(self.n_topics):
            topic_indexes = [i for i,t in enumerate(topics) for p in t if topic in p]
            topic_sentences = [t for i,t in enumerate(self.tokenized_word_sentences) if i in topic_indexes]
            all_topic_words = [word for sentence in topic_sentences for word in sentence]
            if method == "freq":
                frequency = FreqDist(all_topic_words)
                top_n = frequency.most_common(self.topk)
                top_n_words = [t[0] for t in top_n]
            if method == "tfidf":
                topic_data = self.data.iloc[topic_indexes].values
                tfidf = TfidfVectorizer()
                transformed_data = tfidf.fit_transform(topic_data)
                top_word_indexes = np.squeeze(np.asarray(transformed_data.mean(axis=0))).argsort()[-self.topk:]
                print(top_word_indexes)
                wordlist = tfidf.get_feature_names_out()
                top_n_words = [wordlist[i] for i in top_word_indexes]
                print(top_n_words)
            topic_top_n.append(top_n_words)
        
        self.topic_top_n = topic_top_n
        return topic_top_n
    
    def get_top_topic_tokens_lda(self,topics,method="freq"):
        topic_top_n = list()
        for topic in range(self.n_topics):
            topic_indexes = [i for i,t in enumerate(topics) if topic in t]
            topic_sentences = [t for i,t in enumerate(self.tokenized_word_sentences) if i in topic_indexes]
            all_topic_words = [word for sentence in topic_sentences for word in sentence]
            if method == "freq":
                frequency = FreqDist(all_topic_words)
                top_n = frequency.most_common(self.topk)
                top_n_words = [t[0] for t in top_n]
                
            if method == "tfidf":
                topic_data = self.data.iloc[topic_indexes].values
                tfidf = TfidfVectorizer()
                transformed_data = tfidf.fit_transform(topic_data)
                top_word_indexes = np.squeeze(np.asarray(transformed_data.mean(axis=0))).argsort()[-self.topk:]
                wordlist = tfidf.get_feature_names_out()
                top_n_words = [wordlist[i] for i in top_word_indexes]
            topic_top_n.append(top_n_words)
        self.topic_top_n = topic_top_n
        return topic_top_n
    


    def get_topic_diversity(self,top_tokens):
        unique_words = set()
        for topic in range(self.n_topics):
            unique_words = unique_words.union(set(top_tokens[topic]))
        diversity = len(unique_words) / (self.topk * self.n_topics)
        return diversity
    

    def get_coherence(self,top_tokens):
        cm = CoherenceModel(topics=top_tokens,texts = self.tokenized_word_sentences, dictionary=self.id2word, coherence=self.coherence_mode,processes=1,topn=10)
        coherence = cm.get_coherence()
        return coherence
    

    def average_topic_matching(self,english_topics,spanish_topics):
        matches= list()
        for i in range(len(english_topics)):
            english_document_topics = english_topics[i][0]
            spanish_document_topics = spanish_topics[i][0]
            match = np.array_equal(english_document_topics,spanish_document_topics)
            matches.append(match)
        return np.array(matches).mean()
    
    def _get_tokenized_word_sentences(self,texts):
        return [word_tokenize(s) for s in texts]


    def _get_topic_counters(self,tokenized_word_sentences,topics):
        topic_counters = list()
        for topic in range(self.n_topics):
                    topic_indexes = [i for i,t in enumerate(topics) for p in t if topic in p]
                    topic_sentences = [t for i,t in enumerate(tokenized_word_sentences) if i in topic_indexes]
                    all_topic_words = [word for sentence in topic_sentences for word in sentence]
                    frequency = Counter(all_topic_words)
                    topic_counters.append(frequency)
        return topic_counters
    
    def _get_topic_counters_lda(self,tokenized_word_sentences,topics):
        topic_counters = list()
        for topic in range(self.n_topics):
                    topic_indexes = [i for i,t in enumerate(topics) if topic in t]
                    topic_sentences = [t for i,t in enumerate(self.tokenized_word_sentences) if i in topic_indexes]
                    all_topic_words = [word for sentence in topic_sentences for word in sentence]
                    frequency = Counter(all_topic_words)
                    topic_counters.append(frequency)
        return topic_counters
    

    def _calculate_word_probabilities(self,topic_counters,vocab_size,beta=0.01):
        word_dict_probs = list()
        for topic in topic_counters:
            word_probabilities = dict()
            denominator = ( sum(list(topic.values())) + (vocab_size * beta))
            for word,counter in topic.items():
                numerator = counter + beta
                prob = numerator / denominator
                word_probabilities[word] = prob
            
            word_dict_probs.append(word_probabilities)
        return word_dict_probs
    

    def _calculate_word_vectors(self,word_probabilities,vocabulary):
        word_norms = dict()
        word_vectors = dict()
        for word in vocabulary:
            word_norm_total = 0
            for topic,word_topic_probabilities in enumerate(word_probabilities):
                if word in word_topic_probabilities:
                    word_norm_total += word_topic_probabilities[word]
            word_norms[word] = word_norm_total
            word_vector = list()
            for topic,word_topic_probabilities in enumerate(word_probabilities):
                if word in word_topic_probabilities:
                    word_vector.append(word_topic_probabilities[word])
                else:
                    word_vector.append(0)
            word_vector_array = np.array(word_vector) / word_norms[word]
            word_vectors[word] = word_vector_array
        return word_vectors

    def _get_ner(self,texts,language):
        language_model_name_dict = {"english":"en_core_web_sm","spanish":"es_core_news_sm"}
        model = spacy.load(language_model_name_dict[language], disable = ['parser'])
        entities = set()
        for text in texts:
            data = model(text)
            for ent in data.ents:
                entity_text = ent.text
                for entity in entity_text.split(" "):
                    entities.add(entity)
        return entities
    

    def _get_language_vectors_ner(self,text,topics,language,lda=False):
        tokenized_word_sentences = self._get_tokenized_word_sentences(text)
        if lda:
            topic_counters = self._get_topic_counters_lda(tokenized_word_sentences,topics)
        else:
            topic_counters = self._get_topic_counters(tokenized_word_sentences,topics)
        print(topic_counters)
        all_words = [word for sentence in tokenized_word_sentences for word in sentence]
        frequency = Counter(all_words)
        word_probabilities = self._calculate_word_probabilities(topic_counters,len(frequency.keys()))
        print(word_probabilities)
        word_vectors = self._calculate_word_vectors(word_probabilities,list(frequency.keys()))
        print(word_vectors)
        return word_vectors

    def get_cross_lingual_alignment(self,english_topics,spanish_topics,english_text,spanish_text,english_entities,spanish_entities,lda=False):
        import json
        english_vectors = self._get_language_vectors_ner(english_text,english_topics,"english",lda)
        spanish_vectors = self._get_language_vectors_ner(spanish_text,spanish_topics,"spanish",lda)
        common_entities = english_entities.intersection(spanish_entities)
        comparison_vectors = dict()
        for word in common_entities:
            comparison_vectors[word] = dict()
            comparison_vectors[word]["en"] = english_vectors[word].tolist()
            comparison_vectors[word]["es"] = spanish_vectors[word].tolist()
        print(comparison_vectors)
        with open("comparison.json","w") as comp:
            json.dump(comparison_vectors,comp)
        similarities = list()
        for word in common_entities:
            similarity = cosine_similarity(np.array(comparison_vectors[word]["en"]).reshape(1,-1),np.array(comparison_vectors[word]["es"]).reshape(1,-1))
            similarities.append(similarity)
        average_similarity = np.array(similarities).mean()
        print(average_similarity)
        return average_similarity



    def get_dataset_stats(self,dataset):
        stats = dict()
        stats["dataset_length"] = dataset.shape[0]
        stats["average_document_length"] = dataset.str.split().apply(len).median()

        return stats

    

    

