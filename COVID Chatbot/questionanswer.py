from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from textpreprocessing import TextPreprocessing
from bertembedding import BertEbmedding
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


class QuestionAnswerEngine:

    def __init__(self):
        self.df = pd.read_excel("Engine_2/WHO_FAQ (1).xlsx")
        # self.txp = TextPreprocessing(self.df.copy(), 'Context')
        # self.preprocessed_data_df = self.txp.preprocessing()
        # self.bert_embedding = BertEbmedding(self.preprocessed_data_df)
        self.module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        self.model = hub.load(self.module_url)
        print('Init Done')

    def user_question_process(self, question):
        self.test_query_string = question
        print('engine1', self.test_query_string)
        self.test_df = pd.DataFrame(
            [self.test_query_string], columns=['test_questions'])
        print(self.test_df)
        self.test_text_preprocessor = TextPreprocessing(
            self.test_df, 'test_questions')
        self.processed_test_df = self.test_text_preprocessor.preprocessing()
        self.response = self.bert_embedding.bertEmbedding(
            self.test_df, self.processed_test_df)
        print(self.response)
        return self.response

    def universal_sentence_encoder(self, question):
        self.test_query_string = question
        print('USE engine1', self.test_query_string)
        self.all_sentences = [
            row for row in self.df['Context'].values.tolist()]
        self.all_sentences_embedding = self.model(self.all_sentences)
        self.query_embedding = self.model([self.test_query_string])
        self.scores = []
        for embedding in self.all_sentences_embedding:
            self.score = cosine_similarity(embedding.reshape(
                1, -1), self.query_embedding.reshape(1, -1))
            self.scores.append(self.score[0][0])
        return self.df.loc[self.scores.index(max(self.scores)), 'Answer']
