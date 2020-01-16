from cat_bert_model import *

bert_path = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
text_col = 'processed_extract'
label_col = 'scategory'
max_seq_length = 256
epochs = 50
batch_size = 128
split = [0.80, 0.10, 0.10]

report = ''

models = ['models/bert_wiki_4x2131_80test.h5', 'models/bert_wiki_en_84test.h5', 'models/bert_wiki_fr_82test.h5']
data = ['/home/ubuntu/topic_classification/data/wiki_data_labeled_4x2131.csv', '/home/ubuntu/topic_classification/data/wiki_data_labeled_0.csv', '/home/ubuntu/topic_classification/data/wiki_data_fr_labeled_0.csv']
labels = ['gcategory', 'scategory', 'scategory']

for model_name, data_path, label_col in zip(models, data, labels):
    bert_layer, df, X_train, y_train, X_val, y_val, X_test, y_test, n_classes, class_names = load_dataset(bert_path, max_seq_length, data_path, text_col, label_col, split=split)
    model = build_model(bert_layer, max_seq_length, n_classes)
    preds, y, r = evaluate_model(df, class_names, model_name, X_test, y_test)
    report = report + r + '\n\n\n'

with open('results/reports.txt', 'a') as f:
    f.write(report)

