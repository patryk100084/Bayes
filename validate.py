import os
import cv2
import numpy
import time
import math
import multiprocessing
import image_loader

# nessecary directories
metrics_dir = 'metrics'
output_dir = 'output'
ground_truth_dir = 'SkinBin'

# read lines from text files
prob_maps_names = os.listdir(output_dir)

chunk_size = 125
# number of images and masks loaded succesfully
masks_and_maps_loaded = 0
maps_validated = 0

# probability above which pixel is considered skin
thresholds = [0.45]

def validate_classification(thread_number, threshold, map_names):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    counter = 0
    print("THREAD " + str(thread_number) + " INFO: thread started working")
    for i in range(len(map_names)):
        prob_map = image_loader.load_image(output_dir, map_names[i], 1)
        ground_truth_name = map_names[i].replace("_map.jpg", "_s.bmp")
        ground_truth = image_loader.load_image(ground_truth_dir, ground_truth_name, 1)
        if prob_map.shape[0] == ground_truth.shape[0] and prob_map.shape[1] == ground_truth.shape[1]:
            for j in range(prob_map.shape[0]):
                for k in range(prob_map.shape[1]):
                    if prob_map[j][k] > threshold and ground_truth[j][k] == 0.0: # if pixel is classified as skin and it is considered skin in mask
                        TP += 1
                    elif prob_map[j][k] > threshold and ground_truth[j][k] == 1.0: # if pixel is classified as skin and it is considered non-skin in mask
                        FP += 1
                    elif prob_map[j][k] < threshold and ground_truth[j][k] == 0.0: # if pixel is classified as non-skin and it is considered skin in mask
                        FN += 1
                    elif prob_map[j][k] < threshold and ground_truth[j][k] == 1.0: # if pixel is classified as non-skin and it is considered non-skin in mask
                        TN += 1
            counter += 1
            if counter % 25 == 0:
                print("THREAD " + str(thread_number) + " : validated " + str(counter) + " / " + str(len(map_names)) + " images")
        
    print("THREAD " + str(thread_number) + " INFO: thread finished working")
    return [TP,TN,FP,FN] 

if __name__ == '__main__':

    start_time = time.time()

    maps_names_chuncks = [prob_maps_names[i:i+chunk_size] for i in range(0,len(prob_maps_names),chunk_size)]

    for threshold in thresholds:
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        print("--- starting validation ---")
        print("--- threshold:" + str(threshold) + " ---")

        args=[(i,threshold,maps_names_chuncks[i]) for i in range(len(maps_names_chuncks))]
        pool = multiprocessing.Pool()
        results = pool.starmap(validate_classification, args)
        for i in range(len(results)):
            TP += results[i][0]
            TN += results[i][1]
            FP += results[i][2]
            FN += results[i][3] 

        FPR = FP/(FP+TN)
        FNR = FN/(FN+TP)
        recall = TP/(FN+TP)
        precision = TP/(TP+FP)
        Fmeasure = 2*precision*recall/(precision+recall)

        metrics_file_name = "metrics_" + str(threshold) + ".txt"
        metrics_path = os.path.join(metrics_dir,metrics_file_name)

        metrics = open(metrics_path, "w")

        metrics.write("Probability threshold above which pixel is classified as skin: " + str(threshold) + "\n\n")
        metrics.write("True Positive = " + str(TP) + '\n')
        metrics.write("True Negavite = " + str(TN) + '\n')
        metrics.write("False Positive = " + str(FP) + '\n')
        metrics.write("False Negative = " + str(FN) + '\n\n')
        metrics.write("False Positive Rate = " + str(FPR) + '\n')
        metrics.write("False Negative Rate = " + str(FNR) + '\n')
        metrics.write("Recall = " + str(recall) + '\n')
        metrics.write("Precision = " + str(precision) + '\n')
        metrics.write("F-measure = " + str(Fmeasure) + '\n')

        metrics.close()

        print("--- finished validation ---")
        print("TP: " + str(TP) + " TN: " + str(TN) + " FP: " + str(FP) + " FN: " + str(FN))

    end_time = time.time()

    print("finished execution: --- " + str(end_time - start_time) + " seconds ---")