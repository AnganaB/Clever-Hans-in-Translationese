import pandas as pd 


def replace_ner(df):
    masked = []
    p = 0
    for i in df.text.values:
        print(p)
        if len(df.loc[df.text == i]) > 0:
            k = df.loc[df.text == i].entity.values
            l = df.loc[df.text == i].label.values

            for a, b in zip(k, l):
        #         print(a)
        #         print(b)
                i = i.replace(a, "[{}]".format(b))
            masked.append(i)
        else:
            masked.append(i)
        p = p + 1


    df['masked'] = masked

    return df 


def main():

    df1 = pd.read_csv('/raid/data/kay/mounted/mono_de_es_train.tsv', sep='\t')
    df2 = pd.read_csv('/raid/data/kay/mounted/mono_de_es_dev.tsv', sep='\t')
    df3 = pd.read_csv('/raid/data/kay/mounted/mono_de_es_test.tsv', sep='\t')
    df = pd.concat([df1, df2, df3])
    
    dff = replace_ner(df)
    
    dff.to_csv("NER_replaced.csv")
    
if __name__ == '__main__':
    main()
