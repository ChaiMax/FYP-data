import streamlit as st
import numpy as np
import pandas as pd
import cv2
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import imblearn
from string import ascii_letters
import seaborn as sns
import matplotlib.pyplot as plt


#Model evaluation 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import classification_report 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error


#Formating
pd.set_option('display.max_columns', 100)
sns.set()

def app():
    #functions
    def evaluate_model(model,model_name,X_train, y_train,X_test, y_test):
        y_pred = model.predict(X_test)
        proba = model.predict_proba(X_test)
        proba = proba[:,1]

        print("Accuracy on training set: {:.2f} %".format(model.score(X_train, y_train)*100))
        print("Accuracy on test set    : {:.2f} %".format(model.score(X_test, y_test)*100))
        print('********************************************')
        print('Accuracy   = {:.2f} %'. format(accuracy_score(y_test, y_pred)*100))
        print('Precision  = {:.2f} %'. format(precision_score(y_test, y_pred)*100))
        print('Recall     = {:.2f} %'. format(recall_score(y_test, y_pred)*100))
        print('F1         = {:.2f} %'. format(f1_score(y_test, y_pred)*100))

        visual_heatmap(y_test, y_pred,model_name)
        fpr,tpr,auc = visual_TP_FP(y_test, proba,model_name)
        pre,rec = visual_precision_recall(y_test, proba,model_name)
        return fpr,tpr,auc,pre,rec

    def visual_heatmap(y_test, y_pred,model_name='Model_name'):
        cf = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5,5))
        group_names = ['True Neg','False Pos','False Neg','True Pos']
        group_counts = ['{0:0.0f}'.format(value) for value in
                    cf.flatten()]
        group_percentages = ['{0:.2%}'.format(value) for value in
                        cf.flatten()/np.sum(cf)]
        labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
                zip(group_names,group_counts,group_percentages)]
        labels = np.asarray(labels).reshape(2,2)

        accuracy  = np.trace(cf) / float(np.sum(cf))
        precision = cf[1,1] / sum(cf[:,1])
        recall    = cf[1,1] / sum(cf[1,:])
        f1_score  = 2*precision*recall / (precision + recall)
        stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(accuracy,precision,recall,f1_score)

        sns.heatmap(cf, annot=labels,fmt ='')
        plt.title(model_name + ' Model Evaluation')
        plt.xlabel('Predicted label'+ stats_text)
        plt.ylabel('Ground truth label')
        st.pyplot()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        
    def visual_TP_FP(y_test, proba,model_name='model_name'):
        plt.figure(figsize=(5,5))
        fpr, tpr, thresholds = roc_curve(y_test, proba) 
        auc = roc_auc_score(y_test, proba)
        plt.plot(fpr,tpr,label = 'AUC = %0.2f' % auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.title('ROC Curve for ' + model_name)
        plt.show()
        st.pyplot()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        return fpr,tpr,auc

    def visual_precision_recall(y_test, proba,model_name='model_name'):
        plt.figure(figsize=(5,5))
        precision, recall, thresholds = precision_recall_curve(y_test, proba) 
        plt.plot(precision, recall,label = model_name)
        plt.legend(loc = 'lower left')
        plt.plot([1, 0], [0.1, 0.1], color='green', linestyle='--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve for ' + model_name)
        plt.show()
        st.pyplot()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        return precision, recall

    ################start
    cleaned_data = pd.read_csv('cleaned.csv')
    encode_df = pd.read_csv('encoded.csv')
    encode_df.drop('Unnamed: 0',axis=1,inplace=True)
    #t.write(encode_df)

    testdata1 = encode_df.copy()
    to_drop=['symptom','PM10','NO2','O3','temperature','DTR','RH','SCORAD']
    X = testdata1.drop(to_drop,1)
    y = testdata1.symptom
    colnames = X.columns

    X = X.drop('Grouped_rainfall',axis = 'columns')
    X = X.drop('DOW',axis = 'columns')

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.30, random_state=10)
    print('\nTo train features: ', len(X.columns))
    print('List of To train features :\n',X.columns)

    print("\nThe number of rows in the train set (Without SMOTE)  : ", X_train.shape)
    print("The number of rows in the  test set (Without SMOTE)  : ", X_test.shape)

    smt = imblearn.over_sampling.SMOTE(sampling_strategy="minority", random_state=10, k_neighbors=5)
    X_res, y_res = smt.fit_resample(X, y)
    X_trainS, X_testS, y_trainS, y_testS = train_test_split(X_res, y_res, test_size = 0.30, random_state = 10)

    #from sklearn.tree import DecisionTreeClassifier 
    from sklearn.metrics import classification_report 
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import mean_absolute_error

    st.title('Best Machine Learning result')

    st.subheader('Random Forest Classifier With SMOTE')
    rf_final = RandomForestClassifier(max_depth=16, random_state = 10)
    rf_final.fit(X_trainS, y_trainS)
    fpr_rfS,tpr_rfS,auc_rfS,pre_rfS,rec_rfS = evaluate_model(rf_final,'Random Forest with SMOTE',X_trainS, y_trainS,X_testS, y_testS)
    #st.pyplot()
    #st.set_option('deprecation.showPyplotGlobalUse', False)

    import pickle
    pkl_filename = "random_forest_model_SMOTE.pkl"
    #with open(pkl_filename, 'wb') as file:
    #    pickle.dump(rf_final, file)

    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)
    #st.write(X_test)

    score = pickle_model.score(X_test, y_test)
    st.write("Test score: {0:.2f} %".format(100 * score))
    st.set_option('deprecation.showPyplotGlobalUse', False)
    Ypredict = pickle_model.predict(X_test)
    st.write(Ypredict)