import re
import nltk
from nltk import word_tokenize
import spacy
# stopwords
from gensim.parsing.preprocessing import remove_stopwords
# lemma functionality provide by NLTK
from nltk.stem import WordNetLemmatizer
# make sure you downloaded model for lemmatization
nltk.download('wordnet')
# make sure you downloaded model for tokenization
nltk.download('punkt')
nlp = nlp = spacy.load("en_core_web_sm")


class TextPreprocessing():
    def __init__(self, data_df, column_name=None):
        self.data_df = data_df
        self.column_name = column_name
        self.processed_column_name = f"processed_{self.column_name}"

    def convert_lowercase(self):
        self.data_df.fillna('', inplace=True)
        self.data_df[self.column_name] = self.data_df[self.column_name].apply(
            lambda column: column.lower())
#         self.data_df = self.data_df.apply(lambda column: column.astype(str).str.lower(), axis=0)

    def remove_special_symbol(self):
        pattern = '[^\w\s]'
        self.data_df[self.column_name] = self.data_df[self.column_name].apply(
            lambda row: re.sub(pattern, ' ', row))

    def remove_stopwords(self):
        for idx, question in enumerate(self.data_df[self.column_name]):
            self.data_df.loc[idx, self.processed_column_name] = remove_stopwords(
                question)

    def apply_lemmatization(self):
        lemma = WordNetLemmatizer()
        for idx, question in enumerate(self.data_df[self.processed_column_name]):
            lemmatized_sentences = []
            doc = nlp(question.strip())
#             print(doc)
            for word in doc:
                lemmatized_sentences.append(word.lemma_)
            self.data_df.loc[idx, self.processed_column_name] = " ".join(
                lemmatized_sentences)

    def preprocessing(self):
        self.convert_lowercase()
        self.remove_special_symbol()
        self.remove_stopwords()
        self.apply_lemmatization()
        print('preprocessing done')
        return self.data_df
