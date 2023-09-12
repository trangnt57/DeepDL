'''
Hyper parameters based on the Qiu et al. - Deep Just-In-Time Defect Localization

epochs: 50
batch size: 16
optimizer: SGD
initial learning rate: 0.1 -> 0.01
learning rate decay: 0.99
momentum: 0.5
clip gradients norm: 5
loss function: cross entropy loss
validation set: 10% of the training set
saving frequency: every epoch
final model selection: lowest entropy score
'''
import argparse
import csv
import json
import os
import sys
import pandas
from tokenizers import Tokenizer
import tensorflow as tf
from tensorflow.keras.models import load_model

# from deepdl import DeepDLConfig, DeepDLTransformer, DeepDL
# from utils import PlotType, getpd, convert_to_dataframe, plot
from DeepDL import *
import warnings

EPOCHS = 10
BATCH_SIZE = 16
CEN_SEQ_LEN = 160
CON_SEQ_LEN = 4 * CEN_SEQ_LEN
DUMMY_DATA = tf.constant([[0]])

warnings.filterwarnings("ignore")

def generate_data(data_file, batch_size, epochs):
    df = pandas.read_csv(data_file, encoding='utf-8', nrows = 150000)
    tokenizer = Tokenizer.from_file("resource/tokenizer.json")
    i = 0
    ep = 0
    while ep < epochs:
        i = 0
        while i < df.shape[0]:
            cen_lines = []
            con_line_blocks = []
            labels = []
            for b in range(batch_size):
                if i < df.shape[0]:
                    central_line = tokenizer.encode(str(df.iloc[i]["raw_changed_line"])).ids
                    central_line.append(tokenizer.token_to_id("[EOL]"))
                    cen_lines.append(central_line)
                    context_lines = tokenizer.encode(str(df.iloc[i]["forward"]) + "\n" + str(df.iloc[i]["backward"])).ids
                    context_lines.insert(0,tokenizer.token_to_id("[SOS]"))
                    context_lines.append(tokenizer.token_to_id("[EOS]"))
                    con_line_blocks.append(context_lines)
                    labels.append(df.iloc[i]["label"])
                    i += 1
            cen_enc_in, con_enc_in, dec_in, dec_out = preprocess(cen_lines,
                                                                 con_line_blocks)
            yield [cen_enc_in, con_enc_in, dec_in], dec_out
        ep += 1


def train(vocab_size: int, d_fn: str, batch_size : int, epochs: int,
          o_fp: str = None) -> None:
    #     if o_fp == None:
    #         o_fp = os.path.join(getpd(), 'out', 'weights', d_fn.split(os.sep)[-2])

    #     os.makedirs(o_fp, exist_ok=True)

    #cen_lines, con_line_blocks, _ = load_data(d_fn)
    data = pandas.read_csv(d_fn)
    num_data = 150000
    num_batches = num_data//batch_size
    del data
    with tf.distribute.MirroredStrategy().scope():
        #cen_enc_in, con_enc_in, dec_in, dec_out = preprocess(cen_lines,
        #                                                    con_line_blocks)
        config = DeepDLConfig(o_fp, num_data, batch_size)
        model = DeepDLTransformer(vocab_size)

        model.compile(optimizer=config.optimizer, loss=config.loss,
                      metrics=[config.accuracy])
        model.fit_generator(generator=generate_data(d_fn, batch_size, epochs),
                            steps_per_epoch=num_batches,
                            epochs=epochs,
                            callbacks=[config.model_checkpoint])


def load_data_test(fn, _skiprows, _nrows):
    cen_lines = []
    con_line_blocks = []
    labels = []

    tokenizer = Tokenizer.from_file("resource/tokenizer.json")
    df = pandas.read_csv(fn, skiprows = _skiprows, nrows = _nrows)
    df.columns = ['Unnamed: 0', 'commit_id', 'idx', 'changed_type', 'label',
                  'raw_changed_line', 'blame_line', 'line_number', 'index_ctg', 'forward',
                  'backward']
    for idx, row in df.iterrows():
        central_line = tokenizer.encode(df.at[idx, "raw_changed_line"]).ids
        central_line.append(tokenizer.token_to_id("[EOL]"))
        cen_lines.append(central_line)
        context_lines = tokenizer.encode(str(df.at[idx, "forward"]) + "\n" + str(df.at[idx, "backward"])).ids
        context_lines.insert(0,tokenizer.token_to_id("[SOS]"))
        context_lines.append(tokenizer.token_to_id("[EOS]"))
        con_line_blocks.append(context_lines)
        labels.append(df.at[idx, "label"])
    return cen_lines, con_line_blocks, labels


def preprocess(cen_lines: list, con_line_blocks: list) -> tuple:
    eol = cen_lines[0][-1]
    sos = con_line_blocks[0][0]
    eos = con_line_blocks[0][-1]
    padded_cen_lines = []
    padded_con_line_blocks = []
    tokenizer = Tokenizer.from_file("resource/tokenizer.json")
    for i in range(len(cen_lines)):
        if (len(cen_lines[i]) <= CEN_SEQ_LEN
                and len(con_line_blocks[i]) <= CON_SEQ_LEN):
            cen_line = cen_lines[i][:]
            con_line_block = con_line_blocks[i][:]

            for j in range(CEN_SEQ_LEN - len(cen_line)):
                cen_line.append(0)

            padded_cen_lines.append(cen_line)

            for j in range(CON_SEQ_LEN - len(con_line_block)):
                con_line_block.append(0)

            padded_con_line_blocks.append(con_line_block)
    cen_enc_in = tf.constant(padded_cen_lines)
    con_enc_in = tf.constant(padded_con_line_blocks)

    for line in padded_cen_lines:
        idx = line.index(eol)

        line.pop(idx)
        line.insert(idx, eos)

    dec_out = tf.constant(padded_cen_lines)

    for line in padded_cen_lines:
        idx = line.index(eos)

        line.pop(idx)
        line.insert(0, sos)

    dec_in = tf.constant(padded_cen_lines)


    return cen_enc_in, con_enc_in, dec_in, dec_out


def test(vocab_size: int, w_fn: str, d_dn: list, o_fp: str = None) -> None:
    '''
    Tests the trained model of the given weight file
    with the given vocabulary size and all of the test data
    in the given test data directory.
    Top-k accuracy, MAP, MRR of the trained model is printed
    and plots are saved to the given output file path.
    If the output file path is none,
    the plots are saved in out/plots/repository.

    Args:
        vocab_size (int): The vocabulary size.
        w_fn (str): The model weight file name.
        d_dn (str): The test data directory name.
        o_fp (str, optional): The output file path.
    '''


    tokenizer = Tokenizer.from_file("resource/tokenizer.json")
    #     if o_fp == None:
    #         o_fp = os.path.join(getpd(), 'out', 'plots', w_fn.split(os.sep)[-2])

    #     os.makedirs(o_fp, exist_ok=True)
    test_data = pandas.read_csv(d_dn[0])
    num_testing_samples = test_data.shape[0] - 1

    del test_data
    suspicious_scores = []
    sk =  0
    while (sk < num_testing_samples):
        list_cen_lines = []
        list_con_line_blocks = []
        list_labels = []
        print("sk:-------", sk)
        for fn in d_dn:
            cen_lines, con_line_blocks, labels = load_data_test(fn, sk, BATCH_SIZE)
            list_cen_lines.append(cen_lines)
            list_con_line_blocks.append(con_line_blocks)
            list_labels.append(labels)

        sos = list_con_line_blocks[0][0][0]
        eos = list_con_line_blocks[0][0][-1]
        cen_line = list_cen_lines[0][0].copy()

        cen_line.insert(0, sos)
        cen_line.pop()
        with tf.distribute.MultiWorkerMirroredStrategy().scope():
            model = DeepDLTransformer(vocab_size)

            model([tf.constant([list_cen_lines[0][0]]),
                   tf.constant([list_con_line_blocks[0][0]]),
                   tf.constant([cen_line])], training=False)
            model.load_weights(w_fn)
            total_outs = DeepDL(model, sos,eos).evaluate(
                [list_cen_lines, list_con_line_blocks], list_labels)
            for out in total_outs:
                for x in out:
                    suspicious_scores.append(x[-1])
        sk += BATCH_SIZE

    data = pandas.read_csv(d_dn[0], encoding='utf-8')
    data["score"] = suspicious_scores
    data.to_csv("results/deepdl_prediction.csv")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--test_file', type=str)

    args = parser.parse_args()

    train_file = args.train_file
    test_file = args.test_file
    train(vocab_size=30000, d_fn = train_file, batch_size = BATCH_SIZE, epochs = EPOCHS)
    test(vocab_size=30000, w_fn = "results/weights.best_model.hdf5", d_dn = [test_file])
