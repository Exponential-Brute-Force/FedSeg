import json
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

with open(sys.argv[1], 'r') as f:
    lines = f.read().splitlines()

metrics = defaultdict(list)
for line in lines:
    if 'training_acc' in line or 'testing_acc' in line:
        #line = line.replace('INFO:root:', '').replace('size = 5', '').replace('\'', '"')
        line = (line[line.find('{') : line.rfind('}')+1]).replace('\'', '"')
        metric_dict = json.loads(line)
        for key,value in metric_dict.items():
            metrics[key].append(value)

df = pd.DataFrame.from_dict(metrics, orient = 'index').transpose()
df.to_csv(sys.argv[1].replace('.log', '.csv'), index=False)

train_metrics = df.columns[:5]
test_metrics = df.columns[5:]
print(train_metrics, test_metrics)
fig, axs = plt.subplots(5, 2,figsize=(20, 90))
cols = ['ACCURACY', 'ACC_CLASS', 'MEAN IOU', 'FW IOU', 'LOSS']

for i, metric in enumerate(train_metrics):
    y_values = np.array(df[metric].dropna())
    x_values = np.array(range(len(y_values))) * 5
    axs[i, 0].plot(x_values, y_values)
    axs[i, 0].set_title('TRAINING ' + cols[i])

for i, metric in enumerate(test_metrics):
    y_values = np.array(df[metric].dropna())
    x_values = np.array(range(len(y_values)))
    axs[i, 1].plot(x_values, y_values, 'tab:green')
    axs[i, 1].set_title('TEST ' + cols[i])

fig.subplots_adjust(hspace=1.8)
plt.savefig(sys.argv[1].replace('.log', '.png'))
plt.show()