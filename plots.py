import json
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from plotly.subplots import make_subplots
import plotly.graph_objects as go

with open(sys.argv[1], 'r') as f:
    lines = f.read().splitlines()

metrics = defaultdict(list)
for line in lines:
    if 'training_acc' in line or 'testing_acc' in line:
        line = (line[line.find('{') : line.rfind('}')+1]).replace('\'', '"')
        metric_dict = json.loads(line)
        for key,value in metric_dict.items():
            metrics[key].append(value)

for key in metrics:
    metrics[key].pop(0)

df = pd.DataFrame.from_dict(metrics, orient = 'index').transpose()
df.to_csv(sys.argv[1].replace('.log', '.csv'), index=False)

train_metrics = df.columns[:5]
test_metrics = df.columns[5:]
positions = [(1,1), (1,2), (1,3), (2,1), (2,2)]

# Using matplotlib subplots
'''
fig, axs = plt.subplots(5, figsize = (20, 80))
cols = ['ACCURACY', 'ACCURACY_CLASS', 'MEAN IOU', 'FW IOU', 'LOSS']

for i, (train_metric, test_metric, position) in enumerate(zip(train_metrics, test_metrics, positions)):
    print(train_metric, test_metric)
    train_values = np.array(df[train_metric].dropna())
    test_values = np.array(df[test_metric].dropna())
    x_values = np.array(range(len(train_values)))
    axs[i].plot(x_values, train_values, label = 'TRAINING')
    axs[i].plot(x_values, test_values, 'tab:green', label = 'TESTING')
    axs[i].set_title(cols[i])
    axs[i].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=8)

fig.subplots_adjust(hspace=0.2)
plt.savefig(sys.argv[1].replace('.log', '.png'))
'''

# Using subplot2grid
'''
cols = ['ACCURACY', 'ACCURACY_CLASS', 'MEAN IOU', 'FW IOU', 'LOSS']
ax0 = plt.subplot2grid(shape=(3,8), loc=(0,0), colspan=2)
ax1 = plt.subplot2grid((3,8), (0,3), colspan=2)
ax2 = plt.subplot2grid((3,8), (0,6), colspan=2)
ax3 = plt.subplot2grid((3,8), (2,2), colspan=2)
ax4 = plt.subplot2grid((3,8), (2,5), colspan=2)

for i, (train_metric, test_metric) in enumerate(zip(train_metrics, test_metrics)):
    print(train_metric, test_metric)
    train_values = np.array(df[train_metric].dropna())
    test_values = np.array(df[test_metric].dropna())
    x_values = np.array(range(len(train_values)))
    (vars()['ax' + str(i)]).plot(x_values, train_values, label = 'TRAINING')
    (vars()['ax' + str(i)]).plot(x_values, test_values, 'tab:green', label = 'TESTING')
    (vars()['ax' + str(i)]).set_title(cols[i])
ax4.legend(loc='upper center', bbox_to_anchor=(-0.5, -0.2),
          fancybox=True, shadow=True, ncol=8)

fig.subplots_adjust(hspace=0.2)
plt.savefig(sys.argv[1].replace('.log', '.png'))
'''

# Using plotly
fig = make_subplots(
    rows=2, cols=3,
    specs=[[{}, {}, {}],
           [{}, {}, None]],
    subplot_titles=('ACCURACY', 'ACCURACY_CLASS', 'MEAN IOU', 'FW IOU', 'LOSS'),
    vertical_spacing=0.1)

for i, (train_metric, test_metric, position) in enumerate(zip(train_metrics, test_metrics, positions)):
    print(train_metric, test_metric, position)
    flag = False
    if i == 0:
        flag = True
    r, c = position
    train_values = np.array(df[train_metric].dropna())
    test_values = np.array(df[test_metric].dropna())
    x_values = np.array(range(len(train_values)))
    fig.add_trace(go.Scatter(x = x_values, y = train_values, name = 'TRAINING', line_color = 'deepskyblue', showlegend = flag, legendgroup = 'TRAINING'), row = r, col = c)
    fig.add_trace(go.Scatter(x = x_values, y = test_values, name = 'TESTING', line_color = 'green', showlegend = flag, legendgroup='TESTING'), row = r, col = c)
fig.update_layout(height=800, width=1260, margin=dict(t = 45), plot_bgcolor='white')
fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
fig.write_html(sys.argv[1].replace('.log', '.html'))
fig.show()