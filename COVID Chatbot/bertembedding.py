from bert_embedding import BertEmbedding
from retrievesimilarquestion import RetrievSimilarQuestion


class BertEbmedding:

    def __init__(self, preprocessed_train_df):
        self.bert_embedding = BertEmbedding()
        self.QA_questions = preprocessed_train_df["Context"].to_list()
        self.question_QA_bert_embeddings_list = self.bert_embedding(
            self.QA_questions)
        self.preprocessed_train_df = preprocessed_train_df
        # store QA bert embeddings in list
        self.question_QA_bert_embeddings = []
        for embeddings in self.question_QA_bert_embeddings_list:
            self.question_QA_bert_embeddings.append(embeddings[1])

    def bertEmbedding(self, test_df, preprocessed_test_df):
        self.query_QA_questions = test_df["test_questions"].to_list()
        self.query_QA_bert_embeddings_list = self.bert_embedding(
            self.query_QA_questions)

        # store query string bert embeddings in list
        self.query_QA_bert_embeddings = []
        for embeddings in self.query_QA_bert_embeddings_list:
            self.query_QA_bert_embeddings.append(embeddings[1])

        self.bot_response = RetrievSimilarQuestion.retrieve_similar_question(self.question_QA_bert_embeddings, self.query_QA_bert_embeddings,
                                                                             self.preprocessed_train_df, 'Context', preprocessed_test_df, 'test_questions', 'Answer')
        print("bert embedding done")
        return self.bot_response
