import os
import cv2
import numpy as np
import time
import math
import multiprocessing
import image_loader

# nessecary directories
metrics_dir = 'metrics'
skin_maps_dir = 'output\\skin_maps'
ground_truth_dir = 'input\\masks'

# get probability maps from directories
skin_maps_names = os.listdir(skin_maps_dir)

chunk_size = math.ceil(len(skin_maps_names)/multiprocessing.cpu_count())
# number of images and masks loaded succesfully
masks_and_maps_loaded = 0
maps_validated = 0

# threshold for both clasifications
thresholds = [0.5]

def validate_classification(thread_number, threshold, skin_map_names):
    sTP = 0
    sTN = 0
    sFP = 0
    sFN = 0
    print("THREAD " + str(thread_number) + " INFO: thread started working")
    for i in range(len(skin_map_names)):
        skin_prob_map = image_loader.load_image(skin_maps_dir, skin_map_names[i], 1)
        ground_truth_name = skin_map_names[i].replace("_s_map.jpg", "_s.bmp")
        ground_truth = image_loader.load_image(ground_truth_dir, ground_truth_name, 1)
        if skin_prob_map.shape == ground_truth.shape:
            
            # skin classificator validation
            skin_xor = np.logical_xor((skin_prob_map > threshold), ground_truth)
            skin_xnor = ~skin_xor
            FP = np.sum(np.logical_and((skin_prob_map > threshold), ground_truth))
            TN = np.sum(np.logical_and((skin_prob_map <= threshold), ground_truth))
            TP = np.sum(skin_xor) - TN
            FN = np.sum(skin_xnor) - FP

            sTP += TP
            sTN += TN
            sFP += FP
            sFN += FN
        else:
            print("WARNING: " + ground_truth_name + " mask doesn't match original image")
        
    print("THREAD " + str(thread_number) + " INFO: thread finished working")
    return [sTP,sTN,sFP,sFN]

if __name__ == '__main__':

    try:
        os.makedirs(metrics_dir, exist_ok = True)
    except OSError as error:
        pass

    start_time = time.time()

    skin_maps_chuncks = [skin_maps_names[i:i+chunk_size] for i in range(0,len(skin_maps_names),chunk_size)]

    for threshold in thresholds:
        TP = 0
        TN = 0
        FP = 0
        FN = 0


        print("--- starting validation ---")
        print("--- threshold:" + str(threshold) + " ---")

        args=[(i,threshold,skin_maps_chuncks[i]) for i in range(len(skin_maps_chuncks))]
        pool = multiprocessing.Pool()
        results = pool.starmap(validate_classification, args)
        for i in range(len(results)):
            TP += results[i][0]
            TN += results[i][1]
            FP += results[i][2]
            FN += results[i][3]

        Recall = TP/(FN+TP)
        Precision = TP/(TP+FP)

        metrics_file_name = "metrics_" + str(threshold) + ".txt"
        metrics_path = os.path.join(metrics_dir,metrics_file_name)

        metrics = open(metrics_path, "w")

        metrics.write("Probability threshold: " + str(threshold) + "\n\n")
        metrics.write("--- SKIN CLASSIFICATION ---" + "\n\n")
        metrics.write("True Positive = " + str(TP) + '\n')
        metrics.write("True Negavite = " + str(TN) + '\n')
        metrics.write("False Positive = " + str(FP) + '\n')
        metrics.write("False Negative = " + str(FN) + '\n\n')
        metrics.write("False Positive Rate = " + str(FP/(FP+TN)) + '\n')
        metrics.write("False Negative Rate = " + str(FN/(FN+TP)) + '\n')
        metrics.write("Recall = " + str(Recall) + '\n')
        metrics.write("Precision = " + str(Precision) + '\n')
        metrics.write("F-measure = " + str(2*Precision*Recall/(Precision+Recall)) + '\n\n')

        metrics.close()

        print("--- finished validation ---")

    end_time = time.time()

    print("finished execution: --- " + str(end_time - start_time) + " seconds ---")