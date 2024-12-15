import numpy as np
from matplotlib import pyplot as plt


def mask_distribution(labels):
    class_labels = {
        0: 'Background clutter',
        1: 'Building',
        2: 'Road',
        3: 'Tree',
        4: 'Low vegetation',
        5: 'Moving car',
        6: 'Static car',
        7: 'Human'
    }


    flat = np.concatenate([label.ravel() for label in labels])

    ## Classes frequency
    class_counts = np.bincount(flat)

    print('class: frequency : label')
    for cls, freq in enumerate(class_counts):
        print(f'{cls}: {freq}\t: {class_labels[cls]}')


    ## Frequency distribution
    plt.figure(figsize=(12, 5))

    plt.bar(class_labels.values(), class_counts, color='#aaa')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution in Training Data')

    plt.show()

    print()

    ## Pie chart
    plt.figure(figsize=(10, 5))
    plt.pie(class_counts, labels=class_labels.values(), autopct='%1.1f%%',
            startangle=160, colors=plt.cm.Paired.colors)
    plt.title("Class Distribution in the Dataset")
    plt.axis('equal')
    plt.legend(loc='upper right')

    plt.show()
