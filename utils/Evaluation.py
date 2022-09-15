from image_utils import readTif
import numpy as np
import surface_distance as surfdist
import decimal

def confusion_matrix(true_data,class_data):

    TP = (true_data * class_data).sum()

    p0 = (class_data == 0).sum()
    FNandFP = ((true_data + class_data) == 1).sum()
    TN = ((true_data + class_data) == 0).sum()
    FN = FNandFP - p0 + TN
    FP =FNandFP - FN
    conf_m = np.array(((TP,FN),(FP,TN)))
    #print(conf_m.sum())
    oa = (TP + TN)/conf_m.sum()
    pe = ((TP + FP).astype(np.longdouble)*(TP + FN).astype(np.longdouble)+
          (FN + TN).astype(np.longdouble)*(FP + TN).astype(np.longdouble))/\
         (conf_m.sum().astype(np.longdouble)**2)
    kappa = (oa - pe) / (1 - pe)
    return  conf_m,oa,pe,kappa

def cal_miou(conf_m):
    TP, FN,  = conf_m[0]
    FP,TN= conf_m[1]
    miou = (TP / (TP + FP + FN) + TN / (TN + FN + FP)) / 2
    return miou

def cal_F1(conf_m):
    TP, FN,  = conf_m[0]
    FP,TN= conf_m[1]
    Precision = TP/(TP+FP)
    Recall = TP/(TP + FN)
    b = 1
    F1 = (1+ b)*(Precision*Recall)/(b**2*(Precision+Recall))
    return Precision,Recall,F1

if __name__ == '__main__':
    true = r''
    class_i = r''
    true_data = readTif(true)
    class_data = readTif(class_i)
    conf_m,oa,pe,kappa = confusion_matrix(true_data, class_data)
    miou = cal_miou(conf_m)
    Pre, recall,F1_score = cal_F1(conf_m)
    surface_distances = surfdist.compute_surface_distances(true_data.astype(bool), class_data.astype(bool), spacing_mm=(1.0, 1.0))
    #Average symmetric surface distance (ASSD) is a measure of the average of all Euclideandistances between two image volumes.
    # Given the average surface distance (ASD),ASD(X,Y)=minyey d(x,y)
    #where d(x,y) is a 3-D matrix consisting of the Euclidean distances between the two imagevolume
    # s X and Y,ASSD is given as:
    #ASSD(X,Y) = { ASD(X,Y) + ASD(Y,X)}3/ 2
    avg_surf_dist = surfdist.compute_average_surface_distance(surface_distances)
    assd = sum(avg_surf_dist)/len(avg_surf_dist)
    print(conf_m)
    print('oa is {0}, kappa is {1}, miou is {2}'.format(oa,kappa,miou))
    print('Precision is {0}, Recall is {1}, F1 is {2}'.format(Pre,recall,F1_score))
    print('asd is {0}'.format(avg_surf_dist))
    print('assd is {0}'.format(assd))
