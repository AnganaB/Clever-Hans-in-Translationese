import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline


def ner(final_text):
    text_list = list()
    startchar = list()
    endchar = list()
    labellist = list()
    ilist= list()
    score_list = list()

    tokenizer = AutoTokenizer.from_pretrained("mschiesser/ner-bert-german")
    model = AutoModelForTokenClassification.from_pretrained("mschiesser/ner-bert-german")
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)


    entities = []
    labels = []
    texts = []
    startchar = []
    endchar = []
    for i in final_text:
        k = nlp(i)

        for ent in range(len(k)):
            word = k[ent]['word']
            if word.startswith('##'):
                word = entities[len(entities)-1] + word.replace('##','')
                entities.pop()
                labels.pop()
                texts.pop()
                startchar.pop()
                endchar.pop()


            entities.append(word)
            labels.append(k[ent]['entity'])
            startchar.append(k[ent]['start'])
            endchar.append(k[ent]['end'])
            texts.append(i)
            
    data_tuples = list(zip(entities, labels, texts, startchar, endchar))
    df_ = pd.DataFrame(data_tuples, columns=['entity', 'label', 'text', 'start', 'end'])

    df_.drop_duplicates(inplace=True)
    
    return df_ 

def main():
    
    df1 = pd.read_csv('/raid/data/kay/mounted/mono_de_es_train.tsv', sep='\t')
    df2 = pd.read_csv('/raid/data/kay/mounted/mono_de_es_dev.tsv', sep='\t')
    df3 = pd.read_csv('/raid/data/kay/mounted/mono_de_es_test.tsv', sep='\t')

    df = pd.concat([df1, df2, df3])
    final_text = df.text.tolist()
    dff = ner(final_text) 
    
    dff.to_csv("ner_entities_all.csv")

if __name__ == '__main__':
    main()