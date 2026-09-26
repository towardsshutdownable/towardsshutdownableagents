import numpy as np
import matplotlib.pyplot as plt

def train_action_counts(n, action_list):
    if len(action_list) % n:
        raise ValueError("length of action list must be divisible by n")

    group_size = int(len(action_list) / n)
    
    groups = range(n) 
    action_counts = [[],[],[],[],[]]
    for group in groups:
        group_actions = action_list[group*group_size:(group+1)*group_size]
        for number in range(5):
            number_count = group_actions.count(number)
            action_counts[number].append(number_count)

    labels = ["0","1","2","3","4"]  
    colours = [ '#4daf4a', '#dede00', '#f781bf', '#984ea3',  '#e41a1c']

    # Bar width and x locations
    w, x = 0.15, np.arange(n)

    fig, ax = plt.subplots()
    for i, (v, label, colour) in enumerate(zip(action_counts, labels, colours)):
        ax.bar(x + (i - 2) * w, v, width=w, label=label, color = colour)

    fig.set_figheight(4)
    fig.set_figwidth(8)
    plt.figure(dpi=400)

    num_meta_eps = len(action_list) // 32
    meta_eps_per_group = num_meta_eps // n
    print(meta_eps_per_group)
    xticks = []
    for i in range(n):
        xticks.append(i*meta_eps_per_group) 

    ax.set_xticks(x)
    ax.set_xticklabels(xticks)
    ax.set_xlabel(f'Meta-episodes')
    ax.set_ylabel('Frequency')
    ax.legend(title="Action")

    return fig







