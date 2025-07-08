import matplotlib
from save_load import load, save
matplotlib.use('TkAgg', force=True)
from Data_gen import *
from Detection_model import *
from Existing_model import *
from polt_res import *
from Adaptive_Response_Orchestration import *
def full_analysis():
    Datagen()
    X_train_70 = load('X_train_70')
    X_test_30 = load('X_test_30')
    y_train_70 = load('y_train_70')
    y_test_30 = load('y_test_30')
    X_train_80 = load('X_train_80')
    X_test_20 = load('X_test_20')
    y_train_80= load('y_train_80')
    y_test_20= load('y_test_20')




    #Training percentage(70% and 30%)
    #PROPOSED
    met,latency_ms= proposed(X_train_70, X_test_30,y_train_70,y_test_30)
    save('proposed_met_70',met)

    #CNN + TSODE
    met,latency_ms = cnn (X_train_70, X_test_30,y_train_70,y_test_30)
    save('CNN_TSODE_met_70', met)

    #DNN
    cm,latency_ms =dnn(X_train_70, X_test_30,y_train_70,y_test_30)
    save('DNN_met_70',cm)

    #RBFNN + RF
    cm,latency_ms =rbfnn_rf(X_train_70, X_test_30,y_train_70,y_test_30)
    save('RBFNN_RF_met_70', cm)

    #SVM + GA
    cm,latency_ms =svm_ga(X_train_70, X_test_30,y_train_70,y_test_30)
    save('SVM_GA_met_70', cm)


    # Training percentage(80% and 20%)
    # PROPOSED
    met,latency_ms = proposed(X_train_80, X_test_20, y_train_80, y_test_20)
    save('proposed_met_80', met)

    # CNN + TSODE
    met,latency_ms = cnn(X_train_80, X_test_20, y_train_80, y_test_20)
    save('CNN_TSODE_met_80', met)

    # DNN
    cm,latency_ms= dnn(X_train_80, X_test_20, y_train_80, y_test_20)
    save('DNN_met_80', cm)

    # RBFNN + RF
    cm,latency_ms = rbfnn_rf(X_train_80, X_test_20, y_train_80, y_test_20)
    save('RBFNN_RF_met_80', cm)

    # SVM + GA
    cm,latency_ms = svm_ga(X_train_70, X_test_30, y_train_70, y_test_30)
    save('SVM_GA_met_80', cm)



a =0
if a == 1:
    full_analysis()
Reinforcement_Learning()
polt_res()
Response_Latency()