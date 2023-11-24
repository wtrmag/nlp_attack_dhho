from __future__ import division
import numpy as np
import tensorflow as tf
from attack_dhho import HHOAttack
import timeit
import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, LSTM, Bidirectional
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import *

# np.random.seed(3333)
# tf.set_random_seed(3333)
VOCAB_SIZE = 50000
modify_threshold = 0.25
with open('data/aux_files/dataset_%d.pkl' % VOCAB_SIZE, 'rb') as f:
    dataset = pickle.load(f)
with open('data/word_candidates_sense.pkl', 'rb') as fp:
    word_candidate = pickle.load(fp)
with open('data/pos_tags_test.pkl', 'rb') as fp:
    test_pos_tags = pickle.load(fp)

# Prevent returning 0 as most similar word because it is not part of the dictionary
max_len = 250
train_x = pad_sequences(dataset.train_seqs2, maxlen=max_len, padding='post')
train_y = np.array(dataset.train_y)
test_x = pad_sequences(dataset.test_seqs2, maxlen=max_len, padding='post')
test_y = np.array(dataset.test_y)

batch_size = 1
lstm_size = 128
pop_size = 60


def bd_lstm(embedding_matrix):
    max_len = 250
    num_classes = 2
    loss = 'binary_crossentropy'
    activation = 'sigmoid'
    embedding_dims = 300
    num_words = VOCAB_SIZE
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


embedding_matrix = np.load(('data/aux_files/embeddings_glove_%d.npy' % VOCAB_SIZE))
embedding_matrix = embedding_matrix.T
model = bd_lstm(embedding_matrix)
model_path = 'data/bdlstm_models'
model.load_weights(model_path)

test_y2 = np.array([[0, 1] if t == 1 else [1, 0] for t in test_y])
all_scores_origin = model.evaluate(test_x, test_y2)
print('all origin test_loss: %f, accuracy: %f' % (all_scores_origin[0], all_scores_origin[1]))
dhho = HHOAttack(model, word_candidate, dataset,
                 max_iter=20,
                 pop_size=pop_size)
SAMPLE_SIZE = len(dataset.test_y)
TEST_SIZE = 1000
test_idx = np.random.choice(len(dataset.test_y), SAMPLE_SIZE, replace=False)
test_len = []
for i in range(SAMPLE_SIZE):
    # 测试字符串长度
    test_len.append(np.sum(np.sign(test_x[test_idx[i]])))
print('Shortest sentence in our test set is %d words' % np.min(test_len))

test_list = []
orig_list = []
orig_label_list = []
adv_list = []
dist_list = []
adv_orig = []
adv_orig_label = []
fail_list = []
adv_training_examples = []
SUCCESS_THRESHOLD = 0.25
start = timeit.default_timer()
for i in range(SAMPLE_SIZE):
    print('The {}th sentence'.format(i + 1))
    pos_tags = test_pos_tags[test_idx[i]]
    x_orig = test_x[test_idx[i]]
    orig_label = test_y[test_idx[i]]
    orig_preds = model.predict(x_orig[np.newaxis, :])[0]
    if np.argmax(orig_preds) != orig_label:
        print('skipping wrong classified ..')
        print('--------------------------')
        continue
    x_len = np.sum(np.sign(x_orig))
    if x_len >= 100:
        print('skipping too long input..')
        print('--------------------------')
        continue
    if x_len < 10:
        print('skipping too short input..')
        print('--------------------------')
        continue
    print('****** ', len(test_list) + 1, ' ********')
    test_list.append(test_idx[i])
    orig_list.append(x_orig)
    target_label = 1 if orig_label == 0 else 0
    orig_label_list.append(orig_label)
    x_adv = dhho.attack(x_orig, target_label, pos_tags)

    # print('optimize successfully!') if if_delete else print('optimize needed')
    if x_adv is None:
        print('%d failed' % (i + 1))
        fail_list.append(test_idx[i])
    else:
        x_adv = dhho.delete_optimization(x_orig, x_adv, target_label, pos_tags)
        num_changes = np.sum(x_orig != x_adv)
        print('%d word(s) changed.' % (int(num_changes)))
        modify_ratio = num_changes / x_len

        if modify_ratio > modify_threshold:
            print('modified too much:', modify_ratio)
        else:
            print('attack success!')
            adv_list.append(x_adv)
            adv_orig.append(test_idx[i])
            adv_orig_label.append(orig_label)
            adv_training_examples.append(test_idx[i])
            dist_list.append(modify_ratio)

    print('--------------------------')
    if len(test_list) >= TEST_SIZE:
        break
end = timeit.default_timer()
print('Number of successful words:{}, Number of total words:{}'.format(len(adv_list), len(test_list)))
print('Attack success rate : {:.2f}%'.format(len(adv_list) / len(test_list) * 100))
print('Median percentage of modifications: {:.02f}% '.format(np.median(dist_list) * 100))
print('Mean percentage of modifications: {:.02f}% '.format(np.mean(dist_list) * 100))

# with open('result/AD_dhho_sem_IMDB_BiLSTM_{}_{:.02f}%_{:.02f}%_{}s.pkl'
#                   .format(modify_threshold, len(adv_list) / len(test_list) * 100, np.mean(dist_list) * 100, end - start), 'wb') as f:
#     pickle.dump((fail_list, adv_orig_label, adv_orig, adv_list, dist_list, test_list), f)

with open('result2/AD_dhho_IMDB_BiLSTM_{}_{:.02f}%_{:.02f}%_{}s.pkl'
.format(pop_size, len(adv_list) / len(test_list) * 100, np.mean(dist_list) * 100, end - start), 'wb') as f:
    pickle.dump((fail_list, adv_orig_label, adv_orig, adv_list, dist_list, test_list), f)
