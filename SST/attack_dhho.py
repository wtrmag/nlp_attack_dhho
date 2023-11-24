import numpy as np
import copy


def do_replace(x_cur, pos, new_word):
    x_new = x_cur.copy()
    x_new[pos] = new_word
    return x_new


def x_and_y(pop, idx, rand_idx, dim, neighbour_list, rand_idx2=None):
    if rand_idx2 is None:
        if len(neighbour_list[dim]) > 2 and pop[idx][dim] != pop[rand_idx][dim]:
            new_sn = np.random.choice(neighbour_list[dim])
            while new_sn == pop[idx][dim] or new_sn == pop[rand_idx][dim]:
                new_sn = np.random.choice(neighbour_list[dim])
            pop[idx][dim] = new_sn
    else:
        if len(neighbour_list[dim]) > 2 and pop[rand_idx2][dim] != pop[rand_idx][dim]:
            new_sn = np.random.choice(neighbour_list[dim])
            while new_sn == pop[rand_idx2][dim] or new_sn == pop[rand_idx][dim]:
                new_sn = np.random.choice(neighbour_list[dim])
            pop[idx][dim] = new_sn


def x_or_y(pop, idx, another_idx, dim):
    if pop[idx][dim] != pop[another_idx][dim]:
        new_sn = np.random.choice([pop[idx][dim], pop[another_idx][dim]])
        pop[idx][dim] = new_sn


def x_xor_y(pop, idx, another_idx, dim, neighbour_list):
    if pop[idx][dim] != pop[another_idx][dim]:
        new_sn = np.random.choice([pop[idx][dim], pop[another_idx][dim]])
        pop[idx][dim] = new_sn
    else:
        if len(neighbour_list[dim]) > 2:
            new_sn = np.random.choice(neighbour_list[dim])
            while new_sn == pop[idx][dim] or new_sn == pop[another_idx][dim]:
                new_sn = np.random.choice(neighbour_list[dim])
            pop[idx][dim] = new_sn


class HHOAttack(object):
    def __init__(self, model, candidate, dataset, pop_size=60, max_iter=20):
        self.candidate = candidate
        self.invoke_dict = {}
        self.dataset = dataset
        self.dict = self.dataset.dict
        self.inv_dict = self.dataset.inv_dict
        self.model = model
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.temp = 0.3

    def predict_batch(self, sentences):
        return np.array([self.predict(s) for s in sentences])

    """ Select the most effective replacement to word at pos (pos)
            in (x_cur) between the words in replace_list """

    def select_best_replacement(self, pos, x_cur, x_orig, target, replace_list):
        new_x_list = [do_replace(x_cur, pos, w) if x_orig[pos] != w and w != 0 else x_cur for w in replace_list]
        new_x_predicts = self.predict_batch(new_x_list)
        x_scores = new_x_predicts[:, target]
        orig_score = self.predict(x_cur)[target]

        new_x_scores = x_scores - orig_score

        if np.max(new_x_scores) > 0:
            best_id = np.argsort(new_x_scores)[-1]
            if np.argmax(new_x_predicts[best_id]) == target:
                return [1, new_x_list[best_id]]
            return [x_scores[best_id], new_x_list[best_id]]
        return [orig_score, x_cur]

    def perturb(self, x_cur, x_orig, neighbours, w_select_probs, target):
        # Pick a word that is not modified and is not UNK
        x_len = w_select_probs.shape[0]
        rand_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]
        # 尽量选择还未替代过的位置进行替换
        while x_cur[rand_idx] != x_orig[rand_idx] and np.sum(x_orig != x_cur) < np.sum(np.sign(w_select_probs)):
            rand_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]
        replace_list = neighbours[rand_idx]
        return self.select_best_replacement(rand_idx, x_cur, x_orig, target, replace_list)

    def generate_population(self, x_orig, neighbours_list, w_select_probs, target, pop_size):
        pop = []
        pop_scores = []
        for i in range(pop_size):
            tem = self.perturb(x_orig, x_orig, neighbours_list, w_select_probs, target)
            if tem is None:
                return None
            if tem[0] == 1:
                return [tem[1]]
            else:
                pop_scores.append(tem[0])
                pop.append(tem[1])
        return pop_scores, pop

    def predict(self, sentence):
        if tuple(sentence) in self.invoke_dict:
            return self.invoke_dict[tuple(sentence)]
        tem = self.model.predict(np.array([sentence]))[0]
        self.invoke_dict[tuple(sentence)] = tem

        return tem

    def attack(self, x_orig, target, pos_tags):
        self.invoke_dict = {}
        x_adv = x_orig.copy()
        x_len = np.sum(np.sign(x_orig))
        x_len = int(x_len)
        pos_list = ['JJ', 'NN', 'RB', 'VB']

        neighbours_list = []
        for i in range(x_len):
            if x_adv[i] not in range(1, 50000):
                neighbours_list.append([])
                continue
            pair = pos_tags[i]
            if pair[1][:2] not in pos_list:
                neighbours_list.append([])
                continue
            if pair[1][:2] == 'JJ':
                pos = 'adj'
            elif pair[1][:2] == 'NN':
                pos = 'noun'
            elif pair[1][:2] == 'RB':
                pos = 'adv'
            else:
                pos = 'verb'
            if pos in self.candidate[x_adv[i]]:
                neighbours_list.append([neighbor for neighbor in self.candidate[x_adv[i]][pos]])
            else:
                neighbours_list.append([])

        neighbours_len = [len(x) for x in neighbours_list]

        w_select_probs = []
        for pos in range(x_len):
            if neighbours_len[pos] == 0:
                w_select_probs.append(0)
            else:
                w_select_probs.append(min(neighbours_len[pos], 10))
        w_select_probs = w_select_probs / np.sum(w_select_probs)

        orig_score = self.predict(x_orig)
        print('orig', orig_score[target])

        if np.sum(neighbours_len) == 0:
            return None

        # print(neighbours_len)

        tem = self.generate_population(x_orig, neighbours_list, w_select_probs, target, self.pop_size)

        if tem is None:
            return None
        if len(tem) == 1:
            return tem[0]
        pop_scores, pop = tem
        # part_elites = copy.deepcopy(pop)
        # part_elites_scores = pop_scores
        # all_elite_score = np.max(pop_scores)
        top_idx = np.argsort(pop_scores)[-1]
        all_elite = pop[top_idx]

        E_0 = 2 * np.random.random() - 1
        pm = np.random.random()

        for curr_iter in range(self.max_iter):
            print('the {}th iteration'.format(curr_iter + 1))
            E = 2 * E_0 * (1 - curr_iter / self.max_iter)
            for idx in range(self.pop_size):
                for dim in range(x_len):
                    r = np.random.random()
                    e = abs(E)
                    if e > 1:
                        if r < 0.5:
                            rand_idx = np.random.choice(self.pop_size)
                            while rand_idx == idx:
                                rand_idx = np.random.choice(self.pop_size)
                            x_and_y(pop, idx, rand_idx, dim, neighbours_list)
                        else:
                            rand_idx_ls = np.random.choice(self.pop_size, size=2, replace=False)
                            while idx in rand_idx_ls:
                                rand_idx_ls = np.random.choice(self.pop_size, size=2, replace=False)
                            x_and_y(pop, idx, rand_idx_ls[0], dim, neighbours_list, rand_idx_ls[1])
                        flag = 0
                    elif e >= 0.5:
                        rand_idx = np.random.choice(self.pop_size)
                        while rand_idx == idx:
                            rand_idx = np.random.choice(self.pop_size)
                        if r < 0.5:
                            x_or_y(pop, idx, rand_idx, dim)
                        else:
                            x_xor_y(pop, idx, rand_idx, dim, neighbours_list)
                        flag = 1
                    else:
                        if r < 0.5:
                            x_or_y(pop, idx, top_idx, dim)
                        else:
                            x_xor_y(pop, idx, top_idx, dim, neighbours_list)
                        flag = 1

                    if flag == 0:
                        temp = 0.3
                    else:
                        temp = 0.7
                    rand = np.random.random(2)
                    if len(neighbours_list[dim]) != 0 and rand[0] < temp and rand[1] < pm:
                        x = np.random.choice(neighbours_list[dim])
                        if x.size != 0:
                            pop[idx][dim] = x

                curr_score = self.predict(pop[idx])
                print('curr top {}'.format(curr_score[target]))
                if np.argmax(curr_score) == target:
                    return pop[idx]
                pop_scores[idx] = curr_score[target]
                top_idx = np.argsort(pop_scores)[-1]

        return None

    def delete_optimization(self, x_orig, x_adv, target, pos_tags):
        p = np.argwhere((x_orig != x_adv) > 0)
        if len(p) == 0:
            return self.attack(x_orig, target, pos_tags)
        change_list = np.concatenate(p, axis=0)

        # ind_ls = []
        delete_scores = []
        x_delete = copy.deepcopy(x_adv)
        for k in np.random.choice(change_list, len(change_list), replace=False):
            temp = x_delete[k]
            x_delete[k] = x_orig[k]
            delete_score = self.predict(x_delete)
            if np.argmax(delete_score) == target:
                # ind_ls.append(k)
                delete_scores.append(delete_score)
            else:
                x_delete[k] = temp

        if len(delete_scores) != 0:
            return x_delete
        return x_adv
