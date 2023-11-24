import pickle
import numpy as np
import encap_imdb_bert_zy as models
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, LSTM, Bidirectional
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import *

with open('data/aux_files/dataset_50000.pkl', 'rb') as f:
    imdb_dataset = pickle.load(f)

with open('data/word_candidates_sense.pkl', 'rb') as fp:
    word_candidate = pickle.load(fp)

with open('result2/AD_dhho_IMDB_BiLSTM_60_99.00%_6.16%_13062.277062762529s.pkl', 'rb') as f:
    imdb_bilstm_adv = pickle.load(f)

with open('result2/AD_dhho_IMDB_BERT_60_98.00%_5.61%_8819.498031247407s.pkl', 'rb') as f:
    imdb_bert_adv = pickle.load(f)


def bd_lstm(embedding_matrix):
    max_len = 250
    num_classes = 2
    loss = 'binary_crossentropy'
    activation = 'sigmoid'
    embedding_dims = 300
    num_words = 50000
    print('Build word_bdlstm model...')
    model = Sequential()
    model.add(Embedding(  # Layer 0, Start
        input_dim=num_words + 1,  # Size to dictionary, has to be input + 1
        output_dim=embedding_dims,  # Dimensions to generate
        weights=[embedding_matrix],  # Initialize word weights
        input_length=max_len,
        name="embedding_layer",
        trainable=False))
    OPTIMIZER = 'adam'

    model.add(Bidirectional(CuDNNLSTM(128, return_sequences=True)))
    model.add(Bidirectional(CuDNNLSTM(64, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(2, activation=activation))
    model.summary()

    # try using different optimizers and different optimizer configs
    model.compile(OPTIMIZER, loss, metrics=['accuracy'])
    return model


def compute1(transfer, curr):
    curr_label = np.array([0 if x[0] > x[1] else 1 for x in curr[:, 0]])
    predict_label = np.array([0 if x[0] > x[1] else 1 for x in transfer[:, 0]])

    res = np.sum(curr_label == predict_label)
    return res / len(curr_label)


def compute2(transfer, curr):
    orig_label = np.array(curr[1])
    predict_label = np.array([0 if x[0] > x[1] else 1 for x in transfer[:, 0]])

    res = np.sum(orig_label != predict_label)
    return res / len(transfer)


def isEqual(x_ori, x_adv):
    adv = np.array([0 if x[0] > x[1] else 1 for x in x_adv[:, 0]])
    ori = np.array(x_ori)
    return np.sum(ori == adv) == 0


if __name__ == '__main__':
    embedding_matrix = np.load(('data/aux_files/embeddings_glove_%d.npy' % 50000))
    embedding_matrix = embedding_matrix.T
    model1 = bd_lstm(embedding_matrix)
    model_path = 'data/bdlstm_models'
    model1.load_weights(model_path)

    model2 = models.Model(imdb_dataset).cuda()

    transfer1 = np.array([model1.predict(s[np.newaxis, :]) for s in imdb_bert_adv[3]])
    # curr1 = np.array([model2.predict(s[np.newaxis, :]) for s in imdb_bert_adv[3]])
    # f1 = isEqual(imdb_bert_adv[1], curr1)
    # z1 = compute1(transfer1, curr1)
    r1 = compute2(transfer1, imdb_bert_adv[1])

    transfer2 = np.array([model2.predict(s[np.newaxis, :]) for s in imdb_bilstm_adv[3]])
    # curr2 = np.array([model1.predict(s[np.newaxis, :]) for s in imdb_bilstm_adv[3]])
    # f2 = isEqual(imdb_bilstm_adv[1], curr2)
    # z2 = compute1(transfer2, curr2)
    r2 = compute2(transfer2, imdb_bilstm_adv[1])

    print("IMDB")
    print("BERT->BiLSTM:{}".format(1-r1))
    print("BiLSTM->BERT:{}".format(1-r2))
    # with open('result2/transfer.pkl', 'wb') as f:
        # pickle.dump((transfer1, transfer2, curr1, curr2), f)



