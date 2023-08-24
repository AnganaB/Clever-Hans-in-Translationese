import argparse
import torch
import pandas as pd
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoConfig, AutoModelForSequenceClassification
from transformers import BertConfig, BertForSequenceClassification, AdamW, get_scheduler
# from datasets import load_metric
from transformers import TrainingArguments
from transformers import Trainer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.datasets import load_svmlight_file
import gensim
import gensim.models
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from gensim.models import ldamodel
from gensim.matutils import kullback_leibler, jaccard, hellinger, sparse2full
import numpy
import spacy 
import spacy.cli 
import nltk


def process_words(texts, stop_words=stop_words, allowed_tags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    
    nltk.download('stopwords')
    spacy.cli.download("de_core_news_lg")

    nlp = spacy.load('de_core_news_lg', disable=['parser', 'ner'])
    stop_words = nltk.corpus.stopwords.words('german')
    texts = [[word for word in simple_preprocess(str(doc), deacc=False, min_len=2) if word not in stop_words] for doc in texts]
    
    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    
    texts_out = []
    
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc])

        
    texts_out = [[word for word in simple_preprocess(str(doc), deacc=False, min_len=2) if word not in stop_words] for doc in texts_out]    
    
    return texts_out


def corp(data):
    data_ready = process_words(data)
    id2word = corpora.Dictionary(data_ready)

    corpus = [id2word.doc2bow(text) for text in data_ready]

    dict_corpus = {}

    for i in range(len(corpus)):
        for idx, freq in corpus[i]:
            if id2word[idx] in dict_corpus:
                dict_corpus[id2word[idx]] += freq
            else:
                dict_corpus[id2word[idx]] = freq
       
    dict_df = pd.DataFrame.from_dict(dict_corpus, orient='index', columns=['freq'])
    print(dict_df.sort_values('freq', ascending=False).head(10))
    
    id2word.filter_extremes(no_below=10) # remove words below frequency 10 
    corpus = [id2word.doc2bow(text) for text in data_ready]
    
    return corpus
    
    
def lda(df1, df2, df3):
    df = pd.concat([df1, df2, df3], ignore_index=True)
    data = df['text']
    
    corpus = corp(data)
    
    mallet_path = '/content/mallet-2.0.8/bin/mallet'
    ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=10, id2word=id2word)
    
    coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_ready, dictionary=id2word, coherence='c_v')
    coherence_ldamallet = coherence_model_ldamallet.get_coherence()

    pd.set_option('display.max_colwidth', -1)
    topics_df = pd.DataFrame([', '.join([term for term, wt in topic]) for topic in topics], columns = ['Terms per Topic'], index=['Topic'+str(t) for t in range(1, ldamallet.num_topics+1)] )

    
    tm_results = ldamallet[corpus]
    corpus_topics = [sorted(topics, key=lambda record: -record[1])[0] for topics in tm_results]
    df_ = pd.DataFrame(corpus_topics, columns= ['Topic', 'Probability'])
    df_['document'] = df.text
    df_['label'] = df.label
    
    return coherence_ldamallet, topics_df, df_

    
def main():
    df1 = pd.read_csv("mono_de_es_train_pos_tokenized_masked.csv")
    df2 = pd.read_csv("mono_de_es_dev_pos_tokenized_masked.csv")
    df3 = pd.read_csv("mono_de_es_test_pos_tokenized_masked.csv")

    
    coherence, topics, df_prob = lda(df1, df2, df3)
    
    print("Coherence Score:", coherence)
    topics_df.to_csv("Topics.csv")
    df_prob.to_csv("Topics_Prob.csv")

    
# %%time
if __name__ == '__main__':
    main()