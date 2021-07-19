#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

names = ['Farrah', 'Fred', 'Felicia']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
labels = ['Apples', 'Bananas', 'Oranges', 'Peaches']
position = np.zeros(3)

plt.title("Number of Fruit per Person")
plt.ylabel("Quantity of Fruit")
plt.yticks(np.arange(0, 90, step=10))
plt.xticks(np.arange(len(names)), names)
plt.ylim(0, 80)
plt.legend(loc="upper right")

for i in range(0, len(labels)):
    plt.bar(names, fruit[i], width=.5, color=colors[i], label=labels[i], bottom=position)
    position += fruit[i]

plt.show()
