import os, re
import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
from tqdm import tqdm
import numpy as np
from bert.tokenization import FullTokenizer
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report

"""# Tokenize

Next, tokenize our text to create `input_ids`, `input_masks`, and `segment_ids`
"""

class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

def create_tokenizer_from_hub_module(bert_path):
    """Get the vocab file and casing info from the Hub module."""
    bert_layer = hub.KerasLayer(bert_path, trainable=False)
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = FullTokenizer(vocab_file, do_lower_case)

    return tokenizer, bert_layer

def convert_single_example(tokenizer, example, max_seq_length=256):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        input_ids = [0] * max_seq_length
        input_mask = [0] * max_seq_length
        segment_ids = [0] * max_seq_length
        label = 0
        return input_ids, input_mask, segment_ids, label

    tokens_a = tokenizer.tokenize(example.text_a)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0 : (max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids, example.label

def convert_examples_to_features(tokenizer, examples, max_seq_length=256):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    input_ids, input_masks, segment_ids, labels = [], [], [], []
    for example in tqdm(examples, desc="Converting examples to features"):
        input_id, input_mask, segment_id, label = convert_single_example(
            tokenizer, example, max_seq_length
        )
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        labels.append(label)
    return (
        np.array(input_ids),
        np.array(input_masks),
        np.array(segment_ids),
        np.array(labels).reshape(-1, 1),
    )

def convert_text_to_examples(texts, labels):
    """Create InputExamples"""
    InputExamples = []
    for text, label in zip(texts, labels):
        InputExamples.append(
            InputExample(guid=None, text_a=" ".join(text), text_b=None, label=label)
        )
    return InputExamples

# Build model
def build_model(bert_layer, max_seq_length, n_classes):

    act = 'softmax'
    loss = 'categorical_crossentropy'

    in_id = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_ids")
    in_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,  name="input_masks")
    in_segment = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,  name="segment_ids")
    bert_inputs = [in_id, in_mask, in_segment]

    pooled_output, sentence_output = bert_layer(bert_inputs)
    flatten = tf.keras.layers.Flatten()(pooled_output)
    dense_1 = tf.keras.layers.Dense(512, activation='relu')(flatten)
    dropout_1 = tf.keras.layers.Dropout(0.5)(dense_1)
    dense_2 = tf.keras.layers.Dense(256, activation='relu')(dropout_1)
    dense_3 = tf.keras.layers.Dense(128, activation='relu')(dense_2)
    dropout_2 = tf.keras.layers.Dropout(0.4)(dense_3)
    dense_4 = tf.keras.layers.Dense(64, activation='relu')(dropout_2)
    pred = tf.keras.layers.Dense(n_classes, activation=act)(dense_4)

    model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)
    adam = Adam(lr=0.0003)
    model.compile(loss=loss, optimizer=adam, metrics=['accuracy'])
    model.summary()

    return model

def load_dataset(bert_path, max_seq_length, data_path, text_col, label_col, split=[0.80, 0.10, 0.10]):

    df = pd.read_csv(data_path)
    df = df.sample(frac=1).reset_index(drop=True)
    text = df[text_col].tolist()
    texts = [' '.join(t.split()[0:max_seq_length]) for t in text]
    texts = np.array(texts, dtype=object)[:, np.newaxis]
    labels = [1 for i in range(len(text))]
    # instantiate tokenizer and bert model through tf-hub
    print('Instantiating tokenizer and bert model through tf-hub...')
    tokenizer, bert_layer = create_tokenizer_from_hub_module(bert_path)
    # Convert data to InputExample format
    print('Converting inputs...')
    examples = convert_text_to_examples(texts, labels)
    # Convert to features
    (all_input_ids, all_input_masks, all_segment_ids, all_labels
    ) = convert_examples_to_features(tokenizer, examples, max_seq_length=max_seq_length)

    from sklearn.preprocessing import LabelEncoder
    labels = df[label_col].to_list()
    le = LabelEncoder()
    le.fit(labels)
    n_classes = len(list(le.classes_))
    all_labels = le.transform(labels)
    all_labels = tf.keras.utils.to_categorical(all_labels)

    if (np.array(split).sum() != float(1)):
        split = [0.80, 0.10, 0.10]
    else:
        val_size = split[1]
        test_size = split[2]/split[0]

    print('Splitting dataset...')
    train_input_ids, val_input_ids, train_input_masks, val_input_masks, train_segment_ids, val_segment_ids, train_labels, val_labels = train_test_split(all_input_ids, all_input_masks, all_segment_ids, all_labels, test_size=val_size)
    train_input_ids, test_input_ids, train_input_masks, test_input_masks, train_segment_ids, test_segment_ids, train_labels, test_labels = train_test_split(train_input_ids, train_input_masks, train_segment_ids, train_labels, test_size=test_size)

    X_train = [train_input_ids, train_input_masks, train_segment_ids]
    y_train = train_labels
    X_val = [val_input_ids, val_input_masks, val_segment_ids]
    y_val = val_labels
    X_test = [test_input_ids, test_input_masks, test_segment_ids]
    y_test = test_labels

    return bert_layer, df, X_train, y_train, X_val, y_val, X_test, y_test, n_classes, list(le.classes_)

def fit_model(model, X_train, y_train, X_val, y_val, epochs, batch_size):

    model_name = 'models/bert_wiki.h5'
    mcp_save = ModelCheckpoint(model_name, save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.0000000001)
    print('Starting training for {} epochs with a batch size of {}. Saving to {}'.format(epochs, batch_size, model_name))
    history = model.fit(X_train,
                        y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[mcp_save, reduce_lr])

    return history, model_name

def evaluate_model(df, class_names, model_name, X_test, y_test):

    print('Loading model...')
    model = load_model(model_name, custom_objects={'KerasLayer': hub.KerasLayer})
    t1 = tf.convert_to_tensor(X_test[0], dtype=tf.int32)
    t2 = tf.convert_to_tensor(X_test[1], dtype=tf.int32)
    t3 = tf.convert_to_tensor(X_test[2], dtype=tf.int32)
    print('Running predictions...')
    preds = model.predict([t1, t2, t3])
    predictions = np.argmax(preds, axis=1)
    y = np.argmax(y_test, axis=1)
    report = classification_report(y, predictions, target_names=class_names)
    return predictions, y, report
