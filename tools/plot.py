import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

csfont = {'fontname': "Times New Roman",
          'size'    : 22}


# labels = np.load('../CLIPlabel/metadata/pascal/flava_2labels.npy')
# # labels = np.load('metadata/pascal/formatted_train_labels_align.npy')
# labels_gt = np.load('metadata/pascal/formatted_train_labels.npy')
# labels_obs = np.load('metadata/pascal/formatted_train_labels_obs.npy')

exp_dict = np.load('results/20230706_165719/loss_dict.npy', allow_pickle=True)
# loss_pos_false = np.load('results/20230621_174139/pos_false_losses.npy')
# loss_neg_true = np.load('results/20230621_174139/neg_true_losses.npy')
# loss_neg_false = np.load('results/20230621_174139/neg_false_losses.npy')
# observed_means = np.load('results/20230621_174139/observed_means.npy')
# num = np.load('results/20230621_174139/num_reject.npy')
pos_true = []
pos_false = []
neg_true = []
neg_false = []
pos_true_r_num = []
pos_false_r_num = []
neg_true_r_num = []
neg_false_r_num = []
pos_true_p_num = []
pos_false_p_num = []
neg_true_p_num = []
neg_false_p_num = []
bank_mean = []

for i in range(len(exp_dict)):
    pos_true.append(exp_dict[i]['pos_true'].item())
    pos_false.append(exp_dict[i]['pos_false'].item())
    neg_true.append(exp_dict[i]['neg_true'].item())
    neg_false.append(exp_dict[i]['neg_false'].item())
    pos_true_r_num.append(exp_dict[i]['pos_true_r_num'].item())
    pos_true_p_num.append(exp_dict[i]['pos_true_p_num'].item())
    pos_false_r_num.append(exp_dict[i]['pos_false_r_num'].item())
    pos_false_p_num.append(exp_dict[i]['pos_false_p_num'].item())
    neg_true_r_num.append(exp_dict[i]['neg_true_r_num'].item())
    neg_true_p_num.append(exp_dict[i]['neg_true_p_num'].item())
    neg_false_r_num.append(exp_dict[i]['neg_false_r_num'].item())
    neg_false_p_num.append(exp_dict[i]['neg_false_p_num'].item())
    bank_mean.append(exp_dict[i]['bank_mean'].item())

# print(pos_true_r_num)
# print(np.shape(labels_gt))

# loss_pos_true = exp_dict

# labels = np.sum(np.asarray(labels), axis=0)
# labels_gt = np.sum(np.asarray(labels_gt), axis=0)
# # labels_obs = np.sum(np.asarray(labels_obs), axis=0)

# print(labels)
# print(labels_gt)
# # print(labels_obs)

xaxis = np.arange(2860)

fig, ax = plt.subplots(figsize = (12, 7))
# # p1 = plt.bar(xaxis, labels, 0.2)
# # p2 = plt.bar(xaxis + 0.2, labels_gt, 0.2, color='r')
p1 = plt.plot(xaxis, pos_true, 'b:')
p2 = plt.plot(xaxis, neg_true, 'g:')
p3 = plt.plot(xaxis, pos_false, 'r:')
p4 = plt.plot(xaxis, neg_false, 'c:')
p5 = plt.plot(xaxis, bank_mean, 'k--')

# plt.subplot(2, 1, 2)
###### num reject
# p6 = plt.plot(xaxis, pos_true_r_num, 'b:')
# p7 = plt.plot(xaxis, pos_false_r_num, 'r:')
# p8 = plt.plot(xaxis, neg_true_r_num, 'g:')
# p9 = plt.plot(xaxis, neg_false_r_num, 'c:')

##### num pass
# p6 = plt.plot(xaxis, pos_true_p_num, 'b:')
# p7 = plt.plot(xaxis, pos_false_p_num, 'r:')
# p8 = plt.plot(xaxis, neg_true_p_num, 'g:')
# p9 = plt.plot(xaxis, neg_false_p_num, 'c:')
# plt.ylim([-2, 25])
# plt.ylabel('label frequency', **csfont)
# plt.yticks(fontsize=12)
# plt.xticks(fontsize=12)
plt.savefig('save.png')