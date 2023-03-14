import matplotlib.pyplot as plt
import numpy as np

models = ('distilBERT-base-uncased', 'BERT-base-uncased', 'BERT-base-cased', 'RoBERTa-base')
y_pos = np.arange(len(models))
performance = np.array([0.711, 0.703, 0.694, 0.624])

fig, ax = plt.subplots()

hbars = ax.barh(y_pos, performance, align='center')
ax.set_yticks(y_pos, labels=models)
ax.invert_yaxis()
ax.set_xlabel('Accuracy')
ax.set_title('Comparison between different models')

ax.bar_label(hbars, fmt='%.2f') 

plt.savefig('bar.png')