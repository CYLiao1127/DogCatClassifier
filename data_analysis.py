import os
import random
import matplotlib.pyplot as plt

random.seed(100)

if __name__ == '__main__':
    img_dir = "data/data/train"
    img_paths = [os.path.join(img_dir, fname) for fname in os.listdir(img_dir) if fname.endswith('.jpg')]

    random.shuffle(img_paths)

    dog, cat = 0, 0

    for img in img_paths:
        s = img[16:19]
        if s == "dog":
            dog += 1
        elif s == "cat":
            cat += 1

    plt.bar(["dog", "cat"], [dog, cat])

    plt.text(0, dog, str(dog), ha='center', va='bottom')
    plt.text(1, cat, str(cat), ha='center', va='bottom')

    plt.title("Number of Each Class")

    plt.savefig("plots/data_count")