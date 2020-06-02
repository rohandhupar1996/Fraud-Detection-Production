from sklearn.metrics import confusion_matrix 
from sklearn.metrics import auc
from sklearn.metrics import roc_curve




def calc_confusion_matrix(y_test, y_pred):
    '''Calculate true positives, true negatives, false negatives, true negatives
    @param y_test: the label
    @param y_pred: the class predicted by the model
    '''
    return confusion_matrix(y_test, y_pred).ravel() 
                    
    
def evaluate_model(y_test, y_pred, error):
    '''Computes the confusion matrix, precision, recall F1-Score and AUC 
    
    @param y_test: the label
    @param y_pred: the class predicted by the model
    @param pos_label_scores_: prediction scores for the class where customers leave the bank 
    '''
    
    tn, fp, fn, tp=calc_confusion_matrix(y_test, y_pred)
    
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    f1=2*precision*recall/(precision+recall)
    
    fpr, tpr, thresholds = roc_curve(y_test, error)
    roc_auc = auc(fpr, tpr)

    
    print('Precision: %.5f: '%precision)
    print('Recall: %.5f: '%recall)
    print('F1: %.5f: '%f1)
    print('AUC: %f'%roc_auc)
    print(confusion_matrix(y_test, y_pred))  