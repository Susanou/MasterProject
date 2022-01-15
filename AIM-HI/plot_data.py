import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("outputs/plotdata_30.csv")

learners = []
colors = ['b', 'g', 'r', 'c', 'm', 'y']


for i in range(5):
    learners.append(df.loc[df['Learner'] == i, ['Epoch', 'Loss', 'Accuracy']])

ax = learners[0].plot(x='Epoch', y=['Loss', 'Accuracy'], color=colors[0], label=['Loss 0', 'Acc 0'], subplots=True)

for i, l in enumerate(learners[1:]):
    l.plot(x='Epoch', y=['Loss', 'Accuracy'], color=colors[i+1], label=[f'Loss {i+1}', f'Acc {i+1}'], ax=ax, subplots=True)

plt.show()