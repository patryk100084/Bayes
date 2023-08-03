import os
import cv2
import numpy as np
import time
import math
import multiprocessing
import image_loader

# nessecary directories
img_dir = 'input'
output_dir = 'output'

skin_bins_path = "skin_bins.npy"
nonskin_bins_path = "nonskin_bins.npy"
pixel_counts_path = "pixel_counts.npy"

# text files with names of images and their masks used to test bayes classifier
test_file = open("train.txt", "r")

# read lines from text files
test_names = test_file.readlines()

chunk_size = 125

# number of images classified during testing:
test_images_processed = 0
    
def classify_image(thread_number, image_names, skin_probs, nonskin_probs, bin_size, pixel_counts):
    PCs = pixel_counts[0]/(pixel_counts[0]+pixel_counts[1])
    PCns = pixel_counts[1]/(pixel_counts[0]+pixel_counts[1])
    counter = 0
    b_ratio = bin_size[0]/256
    g_ratio = bin_size[1]/256
    r_ratio = bin_size[2]/256
    print("THREAD " + str(thread_number) + " INFO: thread started working")
    for i in range(len(image_names)):
        image_name = image_names[i] + ".jpg"
        test_image = image_loader.load_image(img_dir, image_name, 3)
        skin_prob_map = np.zeros((test_image.shape[0], test_image.shape[1], 1), np.float32)
        for j in range(test_image.shape[0]):
            for k in range(test_image.shape[1]):
                b = math.floor(test_image[j][k][0]*b_ratio)
                g = math.floor(test_image[j][k][1]*g_ratio)
                r = math.floor(test_image[j][k][2]*r_ratio)
                PvCs = skin_probs[b,g,r]
                PvCns = nonskin_probs[b,g,r]
                if (PvCs*PCs + PvCns*PCns) == 0.0: # division by zero
                    skin_prob_map[j][k] = 0.0
                else: 
                    skin_prob_map[j][k] = PvCs*PCs / (PvCs*PCs + PvCns*PCns)
        skin_map_name = image_names[i] + "_s_map.jpg"
        skin_output_path = os.path.join(output_dir, "skin_maps", skin_map_name)
        cv2.imwrite(skin_output_path, np.multiply(skin_prob_map,255))
        counter += 1
        if counter % 25 == 0:
            print("THREAD " + str(thread_number) + " : classified " + str(counter) + " / " + str(len(image_names)) + " iamges successfully")
    print("THREAD " + str(thread_number) + " INFO: thread finished working")
    return counter

if __name__ == '__main__':

    start_time = time.time()
    try:
        os.makedirs(os.path.join(output_dir, "skin_maps"), exist_ok = True)
    except OSError as error:
        pass

    for i in range(len(test_names)):
        test_names[i] = test_names[i].strip()

    if not os.path.isfile(skin_bins_path) or not os.path.isfile(nonskin_bins_path) or not os.path.isfile(pixel_counts_path):
        print("ERROR - missing all required .npy files to classify images")
        exit()
    else:
        # load bins from file
        nonskin_bins = np.load(nonskin_bins_path)
        skin_bins = np.load(skin_bins_path)
        bin_size = None

        if skin_bins.shape == nonskin_bins.shape:
            bin_size = skin_bins.shape
        else:
            print("ERROR - non-skin and skin histograms have different dimensions")
            exit()

        # load and assign probabilities from file
        pixel_counts = np.load(pixel_counts_path)

        skin_probs = np.divide(skin_bins, pixel_counts[0])
        nonskin_probs = np.divide(nonskin_bins, pixel_counts[1])
    
        test_names_chuncks = [test_names[i:i+chunk_size] for i in range(0,len(test_names),chunk_size)]

        args=[(i,test_names_chuncks[i],skin_probs,nonskin_probs,bin_size,pixel_counts) for i in range(len(test_names_chuncks))]
        pool = multiprocessing.Pool()
        results = pool.starmap(classify_image, args)
        for i in range(len(results)):
            test_images_processed += results[i]

        print("classified " + str(test_images_processed) + " images")

        end_time = time.time()

        print("finished execution: --- " + str(end_time - start_time) + " seconds ---")