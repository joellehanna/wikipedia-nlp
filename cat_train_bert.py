from cat_bert_model import *

bert_path = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
# data_path = '/home/ubuntu/topic_classification/data/movies_genres_en.csv'
# text_col = 'plot'
#data_path = '/home/ubuntu/topic_classification/data/wiki_data_labeled_4x2131.csv'
data_path = '/home/ubuntu/topic_classification/data/wiki_data_labeled_0.csv'
text_col = 'processed_extract'
label_col = 'scategory'
max_seq_length = 256
epochs = 50
batch_size = 128
split = [0.80, 0.10, 0.10]

bert_layer, df, X_train, y_train, X_val, y_val, X_test, y_test, n_classes, class_names = load_dataset(bert_path, max_seq_length, data_path, text_col, label_col, split=split)
model = build_model(bert_layer, max_seq_length, n_classes)
#history, model_name = fit_model(model, X_train, y_train, X_val, y_val, epochs, batch_size)
model_name = 'models/bert_wiki.h5'
preds, y, report = evaluate_model(df, class_names, model_name, X_test, y_test)
