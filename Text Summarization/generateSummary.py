import numpy as np
import pandas as pd
import nltk
from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer
from nltk.cluster import KMeansClusterer
from scipy.spatial import distance_matrix


class generateSummary:

    def __init__(self):
        print("loading model ...")
        self.model = SentenceTransformer('stsb-roberta-base')

    ''' creating article into list of sentences '''

    def articles_to_sentecnces(self, article):
        self.sentences = sent_tokenize(article)
        self.sentences = [sentence.strip() for sentence in self.sentences]
        return self.sentences

    ''' creating a dataframe from list of sentences '''

    def create_dataframe(self, sentences):
        self.df = pd.DataFrame(sentences, columns=['Sentences'])
        return self.df

    '''we are using sentences_transformer which is an pre-trined model to create sentence embedding, 
    here pre-trained model from sentence transformer is 'stsb-roberta-base'
    it is recommneded to not perform tokenization, stemming and any stop words removal on sentences 
    as sentence_trasformer creates more contenxtual meaning from input se'''

    def sentence_embedding(self, sentence):
        self.embedding = self.model.encode([sentence])
        return self.embedding[0]

    ''' from sentence_embedding, we perform KMeansClusterr to create clusters, here number of clusters is number 
    of sentences that will be included in our summary.
    in this function, we are calculating centroid of cluster which will be used to calculate the 
    distance between sentence_embedding and centroid'''

    def create_clusters(self, dataframe, num_cluster=4, iterations=25):
        self.X = np.array(dataframe['sentence_embedding'].tolist())
        self.k_means_model = KMeansClusterer(
            num_cluster, distance=nltk.cluster.util.cosine_distance, repeats=iterations, avoid_empty_clusters=True)
        self.sen_clusters = self.k_means_model.cluster(
            self.X, assign_clusters=True)
        dataframe['sentence_assigned_cluster'] = pd.Series(
            self.sen_clusters, index=dataframe.index)
        dataframe['centroid'] = dataframe['sentence_assigned_cluster'].apply(
            lambda x: self.k_means_model.means()[x])
        return dataframe

    ''' function to calculate distance from centrod for each sentence_embedding '''

    def calculate_distance(self, row):
        return distance_matrix([row['sentence_embedding']], [row['centroid'].tolist()])[0][0]
    #     print(distance)

    ''' we will sort the dataframe on distance from centroid and then group it by cluster and create summary '''

    def generate_summary(self, df):
        df = df.sort_values('distance_from_centroid', ascending=True)
        self.summary = ''.join(df.groupby('sentence_assigned_cluster').head(1)[
            'Sentences'].tolist())
        return self.summary

    def main(self, article, num_of_lines_in_summary=4):
        self.sentences = self.articles_to_sentecnces(article)
        self.dataframe = self.create_dataframe(self.sentences)
        self.dataframe['sentence_embedding'] = self.dataframe['Sentences'].apply(
            lambda x: self.sentence_embedding(x))
        self.dataframe = self.create_clusters(
            self.dataframe, num_cluster=num_of_lines_in_summary)
        self.dataframe['distance_from_centroid'] = self.dataframe.apply(
            self.calculate_distance, axis=1)
        self.summary = self.generate_summary(self.dataframe)
        return self.summary
