import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import classification_report, roc_auc_score, auc, roc_curve, f1_score
from scipy import interp

def classification_report_csv(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row_data = list(filter(None, row_data))
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    return dataframe

def multiclass_roc_auc_score(y_true, y_score):
    assert y_true.shape == y_score.shape
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = y_true.shape[1]
    # compute ROC curve and ROC area for each class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # compute macro-average ROC curve and ROC area
    # First aggregate all false probtive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    return roc_auc


def windows(data, size, step):
	start = 0
	while ((start+size) < data.shape[0]):
		yield int(start), int(start + size)
		start += step


def segment_signal_without_transition(data, window_size, step):
	segments = []
	for (start, end) in windows(data, window_size, step):
		if(len(data[start:end]) == window_size):
			segments = segments + [data[start:end]]
	return np.array(segments)


def segment_dataset(X, window_size, step):
	win_x = []
	for i in range(X.shape[0]):
		win_x = win_x + [segment_signal_without_transition(X[i], window_size, step)]
	win_x = np.array(win_x)
	return win_x
    

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return sp.csr_matrix.todense(adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt))


def get_adj(num_node, adj_type):
    '''
    channel seq: Fc5.,Fc3.,Fc1.,Fcz.,Fc2.,Fc4.,Fc6.,C5..,C3..,C1..,Cz..,C2..,C4..,C6..,Cp5.,Cp3.,Cp1.,Cpz.,Cp2.,Cp4.,Cp6.,Fp1.,Fpz.,Fp2.,Af7.,Af3.,Afz.,Af4.,Af8.,F7..,F5..,F3..,F1..,Fz..,F2..,F4..,F6..,F8..,Ft7.,Ft8.,T7..,T8..,T9..,T10.,Tp7.,Tp8.,P7..,P5..,P3..,P1..,Pz..,P2..,P4..,P6..,P8..,Po7.,Po3.,Poz.,Po4.,Po8.,O1..,Oz..,O2..,Iz..
				 0    1    2    3    4    5    6    7    8    9    10   11   12   13   14   15   16   17   18   19   20   21   22   23   24   25   26   27   28   29   30   31   32   33   34   35   36   37   38   39   40   41   42   43   44   45   46   47   48   49   50   51   52   53   54   55   56   57   58   59   60   61   62   63
    '''
    self_link = [(i,i) for i in range(num_node)]
    if (adj_type == 'ng'): # neighboring connections with allover
          neighbor_link = [(1, 2), (1,31),(1,39),(1,8),(1,30),(1,32),(1,9),(1,41),
                           (2, 3), (2,32),(2,9),(2,31),(2,33),(2,10),(2,8),
                           (3, 4), (3,33),(3,10),(3,32),(3,34),(3,11),(3,9),
                           (4, 5), (4,34),(4,11),(4,33),(4,45),(4,12),(4,10),
                           (5, 6), (5,35),(5,12),(5,34),(5,36),(5,13),(5,11),
                           (6, 7), (6,36),(6,13),(6,35),(6,37),(6,14),(6,12),
                           (7, 40),(7,37),(7,14),(7,36),(7,38),(7,42),(7,13),
                           (8, 9), (8,15),(8,41),(8,39),(8,16),(8,45),
                           (9, 10), (9,16),(9,15),(9,17),
                           (10, 11), (10,17),(10,16),(10,18),
                           (11, 12), (11,18),(11,17),(11,19),
                           (12, 13), (12,19),(12,18),(12,20),
                           (13, 14), (13,20),(13,19),(13,21),
                           (14, 21), (14,42),(14,40),(14,46),(14,20),
                           (15, 16), (15,45),(15,48),(15,41),(15,47),(15,49),
                           (16, 17), (16,49),(16,48),(16,50),
                           (17, 18), (17,50),(17,51),(17,49),
                           (18, 19), (18,51),(18,50),(18,52),
                           (19, 20), (19,52),(19,51),(19,53),
                           (20, 21), (20,53),(20,52),(20,54),
                           (21, 46), (21,54),(21,42),(21,53),(21,55),
                           (22, 23), (22,26),(22,25),(22,27),
                           (23, 24), (23,27),(23,26),(23,28),
                           (24, 28), (24,27),(24,29),
                           (25, 26), (25,32),(25,31),(25,33),
                           (26, 37), (26,33),(26,34),(26,32),
                           (27, 28), (27,34),(27,33),(27,35),
                           (28, 29), (28,35),(28,34),(28,36),
                           (29, 36), (29,35),(29,37),
                           (30, 31), (30,39),
                           (31, 32), (31,39),
                           (33, 34),
                           (34, 35),
                           (35, 36),
                           (36, 37),
                           (37, 38), (37,40),
                           (38, 40),
                           (39, 41), (39,43),
                           (40, 42), (40,44),
                           (41, 43), (41,45),
                           (42, 44), (42,46),
                           (43, 45),
                           (44, 46),
                           (45, 47), (45,48),
                           (46, 55), (46,54),
                           (47, 48),
                           (48, 49), (48,56),
                           (49, 50), (49,56),(49,57),
                           (50, 51), (50,57),(50,56),(50,58),
                           (51, 52), (51,58),(51,57),(51,59),
                           (52, 53), (52,59),(52,58),(52,60),
                           (53, 54), (53,60),(53,59),
                           (54, 55), (54,60),
                           (56, 57), (56,61),
                           (57, 58), (57,61),(57,62),
                           (58, 59), (58,62),(58,61),(58,63),
                           (59, 60), (59,63),(59,62),
                           (60, 63), 
                           (61, 62), (61,64),
                           (62, 63), (62,64),
                           (63, 64)]
          neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_link]
          edge = self_link + neighbor_link
          # edge = neighbor_link
          # construct adjacency matrix
          A = np.zeros((num_node, num_node))
          for i, j in edge:
              A[i, j] = 1
              A[j, i] = 1
          adj = normalize_adj(A)

    elif (adj_type == 'dg'): # threshold distance optimal self
        A = np.zeros([num_node, num_node])
        loc = pd.read_csv("./EEG_distance_physionet.csv", index_col=False)
        x = np.array(loc['x(mm)'])
        y = np.array(loc['y(mm)'])
        z = np.array(loc['z(mm)'])
        for m in range(num_node):
            for n in range(num_node):
                if m != n:
                    A[m, n] = np.power(
                        np.power((x[m] - x[n]), 2) + np.power((y[m] - y[n]), 2) + np.power((z[m] - z[n]), 2), -0.5)
        x_loc = np.expand_dims(np.where(A<np.mean(A))[0], axis=1)
        y_loc = np.expand_dims(np.where(A<np.mean(A))[1], axis=1)
        loc = np.append(x_loc,y_loc, axis=1)
        for i,j in loc:
            A[i, j] = 0
        for k in range(num_node):
            A[k,k]=np.mean(A[k])
        adj = normalize_adj(A)
    elif (adj_type == 'sg'):
         A = np.zeros([num_node, num_node])
         loc = pd.read_csv("./EEG_distance_physionet.csv", index_col=False)
         x = np.array(loc['x(mm)'])
         y = np.array(loc['y(mm)'])
         z = np.array(loc['z(mm)'])
         for m in range(num_node):
             for n in range(num_node):
                 if m != n:
                     A[m, n] = np.power(
                         np.power((x[m] - x[n]), 2) + np.power((y[m] - y[n]), 2) + np.power((z[m] - z[n]), 2), -0.5)
         x_loc = np.expand_dims(np.where(A<np.mean(A))[0], axis=1)
         y_loc = np.expand_dims(np.where(A<np.mean(A))[1], axis=1)
         loc = np.append(x_loc,y_loc, axis=1)
         for i,j in loc:
             A[i, j] = 0
         for k in range(num_node):
             A[k,k]=min(A[k])
         adj = normalize_adj(A)

    return adj
