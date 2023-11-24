import pickle
import numpy as np

with open('data/aux_files/dataset_50000.pkl', 'rb') as f:
    imdb_dataset = pickle.load(f)

with open('result2/AD_dhho_IMDB_BERT_100_98.40%_5.15%_8581.679474596865s.pkl', 'rb') as f:
    adv_bert_dataset = pickle.load(f)

with open('result2/AD_dhho_IMDB_BiLSTM_60_89.80%_5.48%_6597.425936200656s.pkl', 'rb') as f:
    adv_bilstm_dataset = pickle.load(f)

l1 = np.argsort(adv_bert_dataset[4])[:50]
for i in l1:
    i_ori = adv_bert_dataset[2][i]
    sen = adv_bert_dataset[3][i]

    orig = imdb_dataset.test_text[i_ori]
    adv = ' '.join([imdb_dataset.inv_full_dict[s] for s in sen if s != 0])
    with open('imdb.txt', 'a') as f:
        f.write(orig)
        f.write('\n')
        f.write(adv)
        f.write('\n')
        f.write('\n')

l2 = np.argsort(adv_bilstm_dataset[4])[:50]
for i in l2:
    i_ori = adv_bilstm_dataset[2][i]
    sen = adv_bilstm_dataset[3][i]

    orig = imdb_dataset.test_text[i_ori]
    adv = ' '.join([imdb_dataset.inv_full_dict[s] for s in sen if s != 0])
    with open('imdb.txt', 'a') as f:
        f.write(orig)
        f.write('\n')
        f.write(adv)
        f.write('\n')
        f.write('\n')


# i = np.argmin(sst_adv_dataset[4])
# i_ori = sst_adv_dataset[2][i]
# sen = sst_adv_dataset[3][i]
#
# orig = sst_dataset.test_text[i_ori]
# adv = ' '.join([sst_dataset.inv_full_dict[s] for s in sen if s != 0])

print("Done")