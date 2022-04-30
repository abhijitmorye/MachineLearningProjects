from sklearn.metrics.pairwise import cosine_similarity


class RetrievSimilarQuestion:
    def retrieve_similar_question(train_tf_vector, test_tf_vector, train_qa_df, train_column_name, test_qa_df, test_column_name, train_answer_column_name):
        for test_index, test_vector in enumerate(test_tf_vector):
            sim_score = -1
            sim_q_index = -1

            for train_index, train_vecor in enumerate(train_tf_vector):
                cos_sim_score = cosine_similarity(
                    train_vecor, test_vector)[0][0]
                # print(cos_sim_score)

                if sim_score < cos_sim_score:
                    sim_score = cos_sim_score
                    sim_q_index = train_index
            print("similarity done", sim_q_index, sim_score)
            if sim_q_index != -1:
                return train_qa_df[train_answer_column_name].iloc[sim_q_index]
            else:
                return "Sorry, I did not understand your question."

            # print("*"*100)
            # print(
            #     f"Test Question --> {test_qa_df[test_column_name].iloc[test_index]}")
            # print(
            #     f"Train Question with similarity --> {train_qa_df[train_column_name].iloc[sim_q_index]}")
            # print(
            #     f"Response Answer --> {train_qa_df[train_answer_column_name].iloc[sim_q_index]}")
            # print("*"*100)
