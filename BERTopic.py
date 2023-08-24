import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from bertopic import BERTopic


sns.set()



def bertopic_(df1, df2, df3):


    df = pd.concat([df1, df2, df3], ignore_index=True)
    docs = df['text']


    topic_model = BERTopic(language="german", verbose=True, calculate_probabilities=True)
    topics, probs = topic_model.fit_transform(docs)

    
    prob = pd.DataFrame(topic_model.probabilities_)
    top = (prob.idxmax(axis=1) + 1)
    df['top'] = top # get the topic a document belongs to the most 
    
    
    return df
    
    
def main():
    
    df1 = pd.read_csv("mono_de_es_train_pos_tokenized_masked.csv")
    df2 = pd.read_csv("mono_de_es_dev_pos_tokenized_masked.csv")
    df3 = pd.read_csv("mono_de_es_test_pos_tokenized_masked.csv")

    
    dff = bertopic_(df1, df2, df3)
    
    
    dff.to_csv("bertopic_topics_275.csv")

    
if __name__ == '__main__':
    main()