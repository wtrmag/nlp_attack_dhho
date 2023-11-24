import pickle
# from data_utils import SSTDataset
import numpy as np

with open('/home/wtr/nlp_attack_dhho/SST/data/aux_files/dataset_13837.pkl', 'rb') as f:
    sst_dataset = pickle.load(f)

with open('result2/AD_dhho_SST_BERT_60_94.20%_8.87%_2203.3225889364257s.pkl', 'rb') as f:
    adv_bert_dataset = pickle.load(f)

with open('result2/AD_dhho_SST_BiLSTM_60_94.10%_10.04%_5681.68867325969s.pkl', 'rb') as f:
    adv_bilstm_dataset = pickle.load(f)

l1 = np.argsort(adv_bert_dataset[4])[:100]
for i in l1:
    i_ori = adv_bert_dataset[2][i]
    sen = adv_bert_dataset[3][i]

    orig = sst_dataset.test_text[i_ori]
    adv = ' '.join([sst_dataset.inv_full_dict[s] for s in sen if s != 0])
    with open('sst.txt', 'a') as f:
        f.write(orig)
        f.write('\n')
        f.write(adv)
        f.write('\n')
        f.write('\n')

l2 = np.argsort(adv_bilstm_dataset[4])[:100]
for i in l2:
    i_ori = adv_bilstm_dataset[2][i]
    sen = adv_bilstm_dataset[3][i]

    orig = sst_dataset.test_text[i_ori]
    adv = ' '.join([sst_dataset.inv_full_dict[s] for s in sen if s != 0])
    with open('sst.txt', 'a') as f:
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