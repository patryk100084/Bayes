import os
import cv2
import numpy
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
    
def classify_image(thread_number, image_names, skin_bins, nonskin_bins, bin_size_b, bin_size_g, bin_szie_r, pixel_counts):
    PCs = pixel_counts[0]/(pixel_counts[0]+pixel_counts[1])
    PCns = pixel_counts[1]/(pixel_counts[0]+pixel_counts[1])
    counter = 0
    print("THREAD " + str(thread_number) + " INFO: thread started working")
    for i in range(len(image_names)):
        image_name = image_names[i] + ".jpg"
        test_image = image_loader.load_image(img_dir, image_name, 3)
        prob_map = numpy.zeros((test_image.shape[0], test_image.shape[1], 1), numpy.float32)
        for j in range(test_image.shape[0]):
            for k in range(test_image.shape[1]):
                b = math.floor(test_image[j][k][0]/(256/bin_size_b))
                g = math.floor(test_image[j][k][1]/(256/bin_size_g))
                r = math.floor(test_image[j][k][2]/(256/bin_szie_r))
                PvCs = (skin_bins[b, g, r]/pixel_counts[0])
                PvCns = (nonskin_bins[b, g, r]/pixel_counts[1])
                if (PvCs*PCs + PvCns*PCns) == 0.0: # division by zero
                    prob_map[j][k] = 0.0
                else: 
                    prob_map[j][k] = PvCs*PCs / (PvCs*PCs + PvCns*PCns)
        map_name = image_names[i] + "_map.jpg"
        output_path = os.path.join(output_dir, map_name)
        cv2.imwrite(output_path, numpy.multiply(prob_map,255))
        counter += 1
        if counter % 25 == 0:
            print("THREAD " + str(thread_number) + " : classified " + str(counter) + " / " + str(len(image_names)) + " iamges successfully")
    print("THREAD " + str(thread_number) + " INFO: thread finished working")
    return counter

if __name__ == '__main__':

    start_time = time.time()

    for i in range(len(test_names)):
        test_names[i] = test_names[i].strip()

    if not os.path.isfile(skin_bins_path) or not os.path.isfile(nonskin_bins_path) or not os.path.isfile(pixel_counts_path):
        print("ERROR - missing all required .npy files to classify images")
        exit()
    else:
        # load bins from file
        nonskin_bins = numpy.load(nonskin_bins_path)
        skin_bins = numpy.load(skin_bins_path)
        bin_size = None

        if skin_bins.shape == nonskin_bins.shape:
            bin_size = skin_bins.shape
        else:
            print("ERROR - non-skin and skin histograms have different dimensions")
            exit()

        # load and assign probabilities from file
        pixel_counts = numpy.load(pixel_counts_path)
    
        test_names_chuncks = [test_names[i:i+chunk_size] for i in range(0,len(test_names),chunk_size)]

        args=[(i,test_names_chuncks[i],skin_bins,nonskin_bins,bin_size[0],bin_size[1],bin_size[2],pixel_counts) for i in range(len(test_names_chuncks))]
        pool = multiprocessing.Pool()
        results = pool.starmap(classify_image, args)
        for i in range(len(results)):
            test_images_processed += results[i]

        print("classified " + str(test_images_processed) + " images")

        end_time = time.time()

        print("finished execution: --- " + str(end_time - start_time) + " seconds ---")