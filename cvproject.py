import cv2
import os

# Classification_results key: {1:1040_1, 2:1040_2, 3:sch_a, 4:sch_b, 5:sch_e_2}
# Move each test folder's contents to testfolder

sift = cv2.xfeatures2d.SIFT_create()
path = 'C:/Users/ea7mcaf/Downloads/testfolder'
classification_results = {}
template_1 = cv2.imread('1040_1_template.png')
template_2 = cv2.imread('1040_2_template.png')
template_3 = cv2.imread('sch_a_template.png')
template_4 = cv2.imread('sch_b_template.png')
template_5 = cv2.imread('sch_e_2_template.png')
keypt1,descriptor1 = sift.detectAndCompute(template_1,None)
keypt2,descriptor2 = sift.detectAndCompute(template_2,None)
keypt3,descriptor3 = sift.detectAndCompute(template_3,None)
keypt4,descriptor4 = sift.detectAndCompute(template_4,None)
keypt5,descriptor5 = sift.detectAndCompute(template_5,None)
threshold = 60

for filename in os.listdir(path):
    form = cv2.imread(path+'/'+filename)
    sift = cv2.xfeatures2d.SIFT_create()
    keypt6,descriptor6 = sift.detectAndCompute(form, None)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches_1 = flann.knnMatch(descriptor1,descriptor6,k=2)
    matches_2 = flann.knnMatch(descriptor2,descriptor6,k=2)
    matches_3 = flann.knnMatch(descriptor3,descriptor6,k=2)
    matches_4 = flann.knnMatch(descriptor4,descriptor6,k=2)
    matches_5 = flann.knnMatch(descriptor5,descriptor6,k=2)
    match_list = [matches_1,matches_2,matches_3,matches_4,matches_5]
    
    for matches in match_list:
        match_number = 0
        for m, n in matches:
            if m.distance<0.7*n.distance:
                match_number += 1
        if match_number>threshold:
            classification_results[filename] = match_list.index(matches)+1
            break
        if matches == matches_5 and match_number<threshold:
            classification_results[filename] = '?'

TP1, TP2, TP3, TP4, TP5 = 0,0,0,0,0
num_ones, num_twos, num_threes, num_fours, num_fives = 0,0,0,0,0
answers = {}
errors = 0

for entry in classification_results:
    if '1040_1' in entry:
        answers[entry] = 1
        num_ones +=1
        if classification_results[entry]==1:
            TP1 +=1
    if '1040_2' in entry:
        answers[entry] = 2
        num_twos +=1
        if classification_results[entry]==2:
            TP2 +=1
    if 'sch_a' in entry:
        answers[entry] = 3
        num_threes +=1
        if classification_results[entry]==3:
            TP3 +=1
    if 'sch_b' in entry:
        answers[entry] = 4
        num_fours += 1
        if classification_results[entry]==4:
            TP4 +=1
    if 'sch_e_2' in entry:
        answers[entry] = 5
        num_fives += 1
        if classification_results[entry]==5:
            TP5 +=1
            
precision_1,recall_1 = float(TP1)/classification_results.values().count(1),float(TP1)/num_ones
precision_2,recall_2 = float(TP2)/classification_results.values().count(2),float(TP2)/num_twos
precision_3,recall_3 = float(TP3)/classification_results.values().count(3),float(TP3)/num_threes
precision_4,recall_4 = float(TP4)/classification_results.values().count(4),float(TP4)/num_fours
precision_5,recall_5 = float(TP5)/classification_results.values().count(5),float(TP5)/num_fives
precision = sum([precision_1,precision_2,precision_3,precision_4,precision_5])/5.0
recall = sum([recall_1,recall_2,recall_3,recall_4,recall_5])/5.0

for entry in classification_results:
    if classification_results[entry] != answers[entry]:
        errors += 1

total = sum([num_ones,num_twos,num_threes,num_fours,num_fives])
accuracy = (total-errors)/float(total)
F1_score = 2 * float(precision*recall)/(precision+recall)