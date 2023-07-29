import os
import cv2
import numpy as np
import time
import math
import multiprocessing
import image_loader

# nessecary directories
metrics_dir = 'metrics'
skin_maps_dir = os.path.join('output', 'skin_maps')
nonskin_maps_dir = os.path.join('output', 'nonskin_maps')
ground_truth_dir = 'SkinBin'

# get probability maps from directories
skin_maps_names = os.listdir(skin_maps_dir)
nonskin_maps_names = os.listdir(nonskin_maps_dir)

chunk_size = 125
# number of images and masks loaded succesfully
masks_and_maps_loaded = 0
maps_validated = 0

# threshold for both clasifications
thresholds = [0.45]

def validate_classification(thread_number, threshold, skin_map_names, nonskin_map_names):
    sTP = 0
    sTN = 0
    sFP = 0
    sFN = 0
    nsTP = 0
    nsTN = 0
    nsFP = 0
    nsFN = 0
    print("THREAD " + str(thread_number) + " INFO: thread started working")
    if len(skin_map_names) == len(nonskin_map_names):
        for i in range(len(skin_map_names)):
            skin_prob_map = image_loader.load_image(skin_maps_dir, skin_map_names[i], 1)
            nonskin_prob_map = image_loader.load_image(nonskin_maps_dir, nonskin_map_names[i], 1)
            ground_truth_name = skin_map_names[i].replace("_s_map.jpg", "_s.bmp")
            ground_truth = image_loader.load_image(ground_truth_dir, ground_truth_name, 1)
            if skin_prob_map.shape == ground_truth.shape and nonskin_prob_map.shape == ground_truth.shape:
                
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

                # non-skin classificator validation
                nonskin_xor = np.logical_xor((nonskin_prob_map > threshold), ground_truth)
                nonskin_xnor = ~nonskin_xor
                TP = np.sum(np.logical_and((nonskin_prob_map > threshold), ground_truth))
                FN = np.sum(np.logical_and((nonskin_prob_map <= threshold), ground_truth))
                TN = np.sum(nonskin_xnor) - TP
                FP = np.sum(nonskin_xor) - FN

                nsTP += TP
                nsTN += TN
                nsFP += FP
                nsFN += FN

                # for j in range(skin_prob_map.shape[0]):
                #     for k in range(skin_prob_map.shape[1]):
                #         if ground_truth[j][k] == 0.0: # skin on ground truth
                #             if skin_prob_map[j][k] > threshold: # skin detected by skin classifier 
                #                 sTP += 1
                #             else: # non-skin detected by skin classifier
                #                 sFN += 1 
                #             if nonskin_prob_map[j][k] > threshold: # non-skin detected by non-skin classifer
                #                 nsFP += 1
                #             else: # skin detected by non-skin classifer
                #                 nsTN += 1
                #         else: # non-skin on ground truth
                #             if skin_prob_map[j][k] > threshold: # skin detected by skin classifier 
                #                 sFP += 1
                #             else: # non-skin detected by skin classifier
                #                 sTN += 1
                #             if nonskin_prob_map[j][k] > threshold: # non-skin detected by non-skin classifer
                #                 nsTP += 1
                #             else: # skin detected by non-skin classifer
                #                 nsFN += 1
        
    print("THREAD " + str(thread_number) + " INFO: thread finished working")
    return [sTP,sTN,sFP,sFN,nsTP,nsTN,nsFP,nsFN]

if __name__ == '__main__':

    try:
        os.makedirs(metrics_dir, exist_ok = True)
    except OSError as error:
        pass

    start_time = time.time()

    skin_maps_chuncks = [skin_maps_names[i:i+chunk_size] for i in range(0,len(skin_maps_names),chunk_size)]
    nonskin_maps_chuncks = [nonskin_maps_names[i:i+chunk_size] for i in range(0,len(nonskin_maps_names),chunk_size)]

    for threshold in thresholds:
        sTP = 0
        sTN = 0
        sFP = 0
        sFN = 0
        nsTP = 0
        nsTN = 0
        nsFP = 0
        nsFN = 0

        print("--- starting validation ---")
        print("--- threshold:" + str(threshold) + " ---")

        args=[(i,threshold,skin_maps_chuncks[i],nonskin_maps_chuncks[i]) for i in range(len(skin_maps_chuncks))]
        pool = multiprocessing.Pool()
        results = pool.starmap(validate_classification, args)
        for i in range(len(results)):
            sTP += results[i][0]
            sTN += results[i][1]
            sFP += results[i][2]
            sFN += results[i][3]
            nsTP += results[i][4]
            nsTN += results[i][5]
            nsFP += results[i][6]
            nsFN += results[i][7]

        sRecall = sTP/(sFN+sTP)
        sPrecision = sTP/(sTP+sFP)

        nsRecall = nsTP/(nsFN+nsTP)
        nsPrecision = nsTP/(nsTP+nsFP)

        metrics_file_name = "metrics_" + str(threshold) + ".txt"
        metrics_path = os.path.join(metrics_dir,metrics_file_name)

        metrics = open(metrics_path, "w")

        metrics.write("Probability threshold: " + str(threshold) + "\n\n")
        metrics.write("--- SKIN CLASSIFICATION ---" + "\n\n")
        metrics.write("True Positive = " + str(sTP) + '\n')
        metrics.write("True Negavite = " + str(sTN) + '\n')
        metrics.write("False Positive = " + str(sFP) + '\n')
        metrics.write("False Negative = " + str(sFN) + '\n\n')
        metrics.write("False Positive Rate = " + str(sFP/(sFP+sTN)) + '\n')
        metrics.write("False Negative Rate = " + str(sFN/(sFN+sTP)) + '\n')
        metrics.write("Recall = " + str(sRecall) + '\n')
        metrics.write("Precision = " + str(sPrecision) + '\n')
        metrics.write("F-measure = " + str(2*sPrecision*sRecall/(sPrecision+sRecall)) + '\n\n')

        metrics.write("--- NON-SKIN CLASSIFICATION ---" + "\n\n")
        metrics.write("True Positive = " + str(nsTP) + '\n')
        metrics.write("True Negavite = " + str(nsTN) + '\n')
        metrics.write("False Positive = " + str(nsFP) + '\n')
        metrics.write("False Negative = " + str(nsFN) + '\n\n')
        metrics.write("False Positive Rate = " + str(nsFP/(nsFP+nsTN)) + '\n')
        metrics.write("False Negative Rate = " + str(nsFN/(nsFN+nsTP)) + '\n')
        metrics.write("Recall = " + str(nsRecall) + '\n')
        metrics.write("Precision = " + str(nsPrecision) + '\n')
        metrics.write("F-measure = " + str(2*nsPrecision*nsRecall/(nsPrecision+nsRecall)) + '\n')

        metrics.close()

        print("--- finished validation ---")

    end_time = time.time()

    print("finished execution: --- " + str(end_time - start_time) + " seconds ---")