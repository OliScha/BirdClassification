from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve, validation_curve, train_test_split
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score, median_absolute_error
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

def plot_confusion_matrix(confmat):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i,
            s=confmat[i, j],va='center', ha='center')
    plt.xlabel('predicted label')
    plt.ylabel('true label')


def convert2OneHotFeatureNames(catFeats,featureNames,X):
    '''
    catFeats:       List, which contains the indices of the nominal features
    featureNames:   List of original featureNames
    X:              2-d Numpy Array containing numerical feature-values before one-hot-encoding
    
    function returns onehotFeatureNames, which are the names of the columns of X after one-hot-encoding
    '''
    nonCatFeatureNames=[f for (i,f) in enumerate(featureNames) if i not in catFeats]
    #print nonCatFeatureNames
    onehotFeatureNames=[]
    for c in catFeats:
        vals=np.unique(X[:,c])
        fname=featureNames[c]
        #print "Values of nominal feature in column %d:  "%(c),vals
        for v in vals:
            onehotFeatureNames.append(fname+"="+str(v))
    onehotFeatureNames.extend(nonCatFeatureNames)
    return onehotFeatureNames

def plot_evaluation_curve(trainSize, trainScore, testScore,xlabel="Number of training samples",ylabel="Accuracy",xscale='linear'):
    train_mean = np.mean(trainScore, axis=1)
    min_train_test=np.min((np.min(trainScore),np.min(testScore)))
    max_train_test=np.max((np.max(trainScore),np.max(testScore)))
    train_std = np.std(trainScore, axis=1)
    test_mean = np.mean(testScore, axis=1)
    test_std = np.std(testScore, axis=1)
    plt.figure(figsize=(14,8))
    #print len(train_mean),train_mean.shape
    #print len(train_)
    plt.plot(trainSize, train_mean,color='blue', marker='o',markersize=5, label='training accuracy')
    plt.fill_between(trainSize,train_mean + train_std,train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(trainSize, test_mean, color='green', linestyle='--',marker='s', markersize=5,label='validation accuracy')
    plt.fill_between(trainSize,test_mean + test_std,test_mean - test_std,alpha=0.15, color='green')
    plt.grid()
    plt.xlabel(xlabel)
    plt.xscale(xscale)
    plt.ylabel(ylabel)
    plt.legend(loc='lower right')
    plt.ylim([min_train_test, max_train_test])

def plot_feature_importances(feature_importances, title, feature_names,std="None"):
    
    # Normalize the importance values 
    feature_importances = 100.0 * (feature_importances / max(feature_importances))
    if std=="None":
        std=np.zeros(len(feature_importances))

    # Sort the values and flip them
    index_sorted = np.flipud(np.argsort(feature_importances))

    # Arrange the X ticks
    pos = np.arange(index_sorted.shape[0]) + 0.5

    # Plot the bar graph
    plt.figure(figsize=(16,10))
    plt.bar(pos, feature_importances[index_sorted], align='center',alpha=0.5,yerr=100*std[index_sorted])
    plt.xticks(pos, feature_names[index_sorted])
    plt.ylabel('Relative Importance')
    plt.title(title)

def determineRegressionMetrics(y_test,y_pred,title=""):
    mse = mean_squared_error(y_test, y_pred)
    mad = mean_absolute_error(y_test, y_pred)
    #rmsle=np.sqrt(mean_squared_error(np.log(y_test+1),np.log(y_pred+1)))
    r2=r2_score(y_test, y_pred)
    med=median_absolute_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred) 
    print(title)
    print("Mean absolute error =", round(mad, 2))
    print("Mean squared error =", round(mse, 2))
    print("Median absolute error =", round(med, 2))
    print("R2 score =", round(r2, 2))
    #print("Root Mean Squared Logarithmic Error =",rmsle)
    print("Explained variance score =", round(evs, 2))

def determineClassificationMetrics(y_test,y_pred,title=""):
    print(title)
    print("Accuracy  = \t",accuracy_score(y_test,y_pred))
    print("F1-Score  = \t",f1_score(y_test,y_pred))
    print("Precision = \t",precision_score(y_test,y_pred))
    print("Recall    = \t",recall_score(y_test,y_pred))
    print("ROC_AUC   = \t",roc_auc_score(y_test,y_pred))

