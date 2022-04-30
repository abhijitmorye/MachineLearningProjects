import pandas as pd
import numpy as np
from collections import Counter
from numpy import mean
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


class SymptomPrediction:
    def __init__(self):
        self.df = pd.read_csv('Engine_1/Cleaned_covid_symptoms_prediction.csv')
        self.feature_columns = [
            x for x in self.df.columns.tolist() if x not in ['corona_result', 'Unnamed: 0']]
        print(self.feature_columns)
        self.feature_values = self.df[self.feature_columns].values.tolist()
        self.target_values = self.df['corona_result'].values.tolist()
        self.counter = Counter(self.target_values)
        print(self.counter)
        self.over = SMOTE(sampling_strategy=0.1)
        self.under = RandomUnderSampler(sampling_strategy=0.5)
        self.steps = [('o', self.over), ('u', self.under)]
        self.pipeline = Pipeline(steps=self.steps)
        self.ub_feature_values, self.ub_target_values = self.pipeline.fit_resample(
            self.feature_values, self.target_values)
        self.ub_counter = Counter(self.ub_target_values)
        print(self.ub_counter)
        self.model = DecisionTreeClassifier(max_depth=10)
        # fit model on the training dataset
        self.model_fit = self.model.fit(
            self.ub_feature_values, self.ub_target_values)

    def predictSymptom(self, questionList):
        test_list = np.array([questionList])
        self.prediction = self.model_fit.predict(test_list)
        print(self.prediction)
        if self.prediction == 0:
            self.msg = 'Our system has predicted that, as of now you do not have covid-19 symtoms, Still we recommend you to stay home, stay safe!'
        else:
            self.msg = '''Our system has predicted that you have COVID-19 sysmptoms, we recommend you to get tested and for further guidance, visit the nearest COVID helpcenter'''
        return self.msg
