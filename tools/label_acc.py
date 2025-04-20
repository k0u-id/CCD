import numpy as np

labels = np.load('../CLIPlabel/metadata/pascal/clip_2labels.npy')
# labels = np.load('metadata/pascal/formatted_train_labels_align.npy')
labels_gt = np.load('metadata/pascal/formatted_train_labels.npy')
# labels_obs = np.load('metadata/pascal/formatted_train_labels_obs.npy')


#### top 1 inferences ####

# falses = np.zeros(20)
# correct = 0

# labels_adjust = np.zeros_like(labels)

# for i in range(len(labels_gt)):
#     if np.sum(labels_gt[i] * labels[i]):
#         correct += 1
#         labels_adjust[i] = labels[i]
#     else:
#         falses += labels[i]

# print(falses)

# print(len(labels_gt))
# print(correct)
# print(np.sum(falses))

# np.save('metadata/formatted_train_labels_adjust.npy', labels_adjust)

#### top k inferences ####

falses = np.zeros(20)
correct = 0

labels_adjust = np.zeros_like(labels)

for i in range(len(labels_gt)):
    for j in range(len(falses)):
        if labels_gt[i,j] * labels[i,j]:
            correct += 1
            labels_adjust[i,j] = labels[i,j]
        else:
            falses[j] += labels[i,j]

print(falses)

print(len(labels_gt))
print(correct)
print(np.sum(falses))