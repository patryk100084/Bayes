import random
import os

img_dir = 'input'

images = os.listdir(img_dir)

numbers = []

for i in range(len(images)):
    numbers.append(i)

train_indexes = sorted(random.sample(numbers, 2000))

train = open("train.txt", "w")

for i in range(len(train_indexes)):
    name = images[train_indexes[i]].replace(".jpg", "")
    train.write(name + '\n')
    images[train_indexes[i]] = '0'

train.close()

test = open("test.txt", "w")

for i in range(len(images)):
    if images[i] != '0':
        name = images[i].replace(".jpg", "")
        test.write(name + '\n')

test.close()
