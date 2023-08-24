import pandas as pd
import argparse
import torch
import pandas as pd
import torch.nn as nn 
from transformers import BertForSequenceClassification, AdamW, BertConfig, BertTokenizer, BertModel
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoConfig, AutoModelForSequenceClassification
from transformers import BertConfig, BertForSequenceClassification, AdamW, get_scheduler
# from datasets import load_metric
from transformers import TrainingArguments
from transformers import Trainer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.datasets import load_svmlight_file
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler



def prep_data(sentences):
    input_ids = []
    attention_masks = []

    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
                            sent,                    
                            add_special_tokens = True,
                            max_length = 512,           
                            pad_to_max_length = True,
                            return_attention_mask = True,  
                            return_tensors = 'pt',     
                            truncation=True
                       )

        input_ids.append(encoded_dict['input_ids'])

        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids_train, dim=0)
    attention_masks = torch.cat(attention_masks_train, dim=0)
    labels = torch.tensor(labels_train)
    
    return input_ids, attention_masks, labels


def data_(df1, df2, df3):

    sentences_train = df1.masked.values
    sentences_dev = df2.masked.values
    sentences_test = df3.masked.values
    labels_train = df1.label.values
    labels_dev = df2.label.values
    labels_test = df3.label.values
    
    return sentences_train, sentences_dev, sentences_test, labels_train, labels_dev, labels_test

def dataset(sentences_train, sentences_dev, sentences_test):
    input_ids_train, attention_masks_train, labels_train = prep_data(sentences_train)
    input_ids_dev, attention_masks_dev, labels_dev = prep_data(sentences_dev)
    input_ids_test, attention_masks_test, labels_test = prep_data(sentences_test)
    
    
    train_dataset = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    val_dataset = TensorDataset(input_ids_dev, attention_masks_dev, labels_dev)
    test_dataset = TensorDataset(input_ids_test, attention_masks_test, labels_test)
        
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    test_size = len(test_dataset)

    batch_size = 16

    train_dataloader = DataLoader(
                train_dataset,  
                sampler = RandomSampler(train_dataset), 
                batch_size = batch_size,

            )

    validation_dataloader = DataLoader(
                val_dataset, 
                sampler = SequentialSampler(val_dataset), 
                batch_size = batch_size 
    )

    test_dataloader = DataLoader(
                test_dataset,
                sampler = SequentialSampler(test_dataset), 
                batch_size = batch_size 
    )
    
    
    return train_dataloader, validation_dataloader, test_dataloader 


def masked_model():

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    special_tokens = {'additional_special_tokens': ["[LOC]", "[ORG]", "[PER]"]} # adding masked tokens 
    tokenizer.add_tokens( ["[LOC]", "[ORG]", "[PER]"])


    model = BertForSequenceClassification.from_pretrained(
        "bert-base-multilingual-uncased", 
        num_labels = 2, 
        output_attentions = False, 
        output_hidden_states = True, 
    )
    
    model.resize_token_embeddings(len(tokenizer))
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model= nn.DataParallel(model)#, device_ids=[0, 1, 2])
    model.to(device)
    
    return model 



def train(train_dataloader, validation_dataloader):
        
    model = masked_model()
    optimizer = AdamW(model.parameters(),
                  lr = 4e-5, 
                  eps = 1e-8 
                )
    epochs = 4 
    
    for epoch_i in range(0, epochs):

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()
        total_train_loss = 0

        model.train()
        output_model = "bert_maskedtrain_{}.pth".format(epoch_i)

        for step, batch in enumerate(train_dataloader):

            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)

                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))


            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()        

            outputs = model(b_input_ids, 
                                 token_type_ids=None, 
                                 attention_mask=b_input_mask, 
                                 labels=b_labels)

            loss = outputs[0]
            logits = outputs[1]

            loss = torch.mean(loss)
            total_train_loss += loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)            

        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, output_model)

        print("")
        print("Running Validation...")

        t0 = time.time()
        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():        
                outputs = model(b_input_ids, 
                                       token_type_ids=None, 
                                       attention_mask=b_input_mask,
                                       labels=b_labels)
                loss = outputs[0]
                logits = outputs[1]

            loss = torch.mean(loss)
            total_eval_loss += loss

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            total_eval_accuracy += flat_accuracy(logits, label_ids)


        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        avg_val_loss = total_eval_loss / len(validation_dataloader)

        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

def test_and_acc(test_dataloader):
    print('Predicting labels for {:,} test sentences...'.format(len(input_ids_test)))

    model.eval()

    predictions , true_labels = [], []

    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, 
                          attention_mask=b_input_mask)

        logits = outputs[0]

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        predictions.append(logits)
        true_labels.append(label_ids)

    print('    DONE.')
    
    flat_predictions = np.concatenate(predictions, axis=0)

    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    flat_true_labels = np.concatenate(true_labels, axis=0)

    acc = accuracy_score(flat_true_labels, flat_predictions)

    return acc 

def main():
    df1 = pd.read_csv("mono_de_es_train_pos_tokenized_masked.csv")
    df2 = pd.read_csv("mono_de_es_dev_pos_tokenized_masked.csv")
    df3 = pd.read_csv("mono_de_es_test_pos_tokenized_masked.csv")
    
    sentences_train, sentences_dev, sentences_test, labels_train, labels_dev, labels_test = data_(df1, df2, df3)
    
    train_dataloader, validation_dataloader, test_dataloader = dataset(sentences_train, sentences_dev, sentences_test)

    train(train_dataloader, validation_dataloader)
    test_and_acc(test_dataloader)
    
    acc = test_and_acc(test_dataloader)
    print("Test accuracy:", acc)
    
    
if __name__ == '__main__':
    main()
    
