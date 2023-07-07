import os
import cv2
import numpy
import time
import math
import multiprocessing
import image_loader

# directories containing images and their binary masks
img_dir = 'input'
mask_dir = 'SkinBin'

# text files with names of images and their masks used to train bayes classifier
train_file = open("train.txt", "r")

# read lines from text files
train_names = train_file.readlines()

chunk_size = 125

# amount of pixels in each class
nonskin_pixels = 0
skin_pixels = 0

# pixels of each class - 1st class - skin, 2nd class non-skin
pixel_counts = []

# number of images processed during training:
train_images_processed = 0

# bins size
bin_size = [64,64,64]

# bins (BGR)
nonskin_bins = numpy.zeros((bin_size[0],bin_size[1],bin_size[2]))
skin_bins = numpy.zeros((bin_size[0],bin_size[1],bin_size[2]))

def train_classifier(thread_number, images_names, bin_size_b, bin_size_g, bin_size_r):
    print("THREAD " + str(thread_number) + " INFO: thread started working")
    skin_bins = numpy.zeros((bin_size[0],bin_size[1],bin_size[2]))
    nonskin_bins = numpy.zeros((bin_size[0],bin_size[1],bin_size[2]))
    skin_pixels = 0
    nonskin_pixels = 0
    counter = 0
    for i in range(len(images_names)):
        image_name = images_names[i] + ".jpg"
        mask_name = images_names[i] + "_s.bmp"
        train_image = image_loader.load_image(img_dir, image_name, 3)
        train_mask = image_loader.load_image(mask_dir, mask_name, 1)
        if train_image.shape != train_mask.shape:
            print("WARNING:" + image_name + " and " + mask_name + " dimensions are diffrent")
            continue
        for j in range(train_image.shape[0]):
            for k in range(train_image.shape[1]):
                if train_mask[j][k] == 0.0:
                    skin_pixels += 1
                    b = math.floor(train_image[j][k][0]/(256/bin_size_b))
                    g = math.floor(train_image[j][k][1]/(256/bin_size_g))
                    r = math.floor(train_image[j][k][2]/(256/bin_size_r))
                    skin_bins[b,g,r] += 1
                else:
                    nonskin_pixels += 1
                    b = math.floor(train_image[j][k][0]/(256/bin_size_b))
                    g = math.floor(train_image[j][k][1]/(256/bin_size_g))
                    r = math.floor(train_image[j][k][2]/(256/bin_size_r))
                    nonskin_bins[b,g,r] += 1
        counter += 1
        if counter % 20 == 0:
            print("THREAD " + str(thread_number) + " : processed " + str(counter) + " / " + str(len(images_names)) + " images")
    print("THREAD " + str(thread_number) + " INFO: thread finished working")
    # show training progress in console
    return [skin_bins, nonskin_bins, skin_pixels, nonskin_pixels, counter]

if __name__ == '__main__':

    start_time = time.time()

    for i in range(len(train_names)):
        train_names[i] = train_names[i].strip()

    train_names_chuncks = [train_names[i:i+chunk_size] for i in range(0,len(train_names),chunk_size)]

    print("training classifier...")
    args = [(i, train_names_chuncks[i], bin_size[0], bin_size[1], bin_size[2]) for i in range(len(train_names_chuncks))]
    pool = multiprocessing.Pool()
    results = pool.starmap(train_classifier, args)
    for i in range(len(results)):
        skin_bins = skin_bins + results[i][0]
        nonskin_bins = nonskin_bins + results[i][1]
        skin_pixels += results[i][2]
        nonskin_pixels += results[i][3]
        train_images_processed += results[i][4] 
    print("finished training classifier")

    print("successfully processed images: " + str(train_images_processed))
    print("found " + str(skin_pixels) + " skin pixels")
    print("found " + str(nonskin_pixels) + " non-skin pixels")

    pixel_counts.append(skin_pixels)
    pixel_counts.append(nonskin_pixels)

    numpy.save("skin_bins", skin_bins)
    numpy.save("nonskin_bins", nonskin_bins)
    numpy.save("pixel_counts", pixel_counts)

    end_time = time.time()

    print("finished execution: --- " + str(end_time - start_time) + " seconds ---")