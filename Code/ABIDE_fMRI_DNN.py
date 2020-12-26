# I define all the functions in this py file, otherwise the random seeds can not be fixed

import  os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
assert tf.__version__.startswith('2.')
from    tensorflow import keras
from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics


from sklearn.metrics import confusion_matrix

import numpy as np

import pdb

def preprocess(x, y):
    '''
    convert data into required data type
    '''
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.int32)
    return x,y

def half_features(X_train, X_test):
    '''
    only keep the 1/4 largest and 1/4 smallest variables in x
    not used in the experimentation, just keep it here
    '''
    x_mean = np.mean(X_train,axis=0)
    n = X_train.shape[1]
    n = int(np.round(n/4.0))
    ind_largest = np.argpartition(x_mean, -n)[-n:]
    ind_smallest = np.argpartition(x_mean*-1, -n)[-n:]
    ind = np.concatenate((ind_largest,ind_smallest))
    ind = np.sort(ind)
    X_train = X_train[:,ind]
    X_test = X_test[:,ind]
    return X_train, X_test

def data_generator(X_data, Y_data,batchsz, n_fold, seed, half):
    '''
    define data generator by regular random sampling (stratified sampling will not be used) for cross validation
    '''
    n_test = int(np.ceil(X_data.shape[0]/5))
    indice = np.arange(X_data.shape[0])
    np.random.seed(seed)
    np.random.shuffle(indice)
    indice = list(indice)
    for i in range(n_fold):
        start = n_test*i
        end = n_test*(i+1)
        if end>len(indice):
            end = len(indice)
        train_indice = indice[:start] + indice[end:]
        test_indice = indice[start:end]
        x = X_data[train_indice]
        y = Y_data[train_indice]
        x_test = X_data[test_indice]
        y_test = Y_data[test_indice]
        if half:
            x,x_test = half_features(x,x_test)
        db = tf.data.Dataset.from_tensor_slices((x,y))
        db = db.map(preprocess).shuffle(x.shape[0]).batch(batchsz)
        #db.batch(batchsz)

        db_test = tf.data.Dataset.from_tensor_slices((x_test,y_test))
        db_test = db_test.map(preprocess).batch(batchsz)
        yield db, db_test
def data_generator_stratify(X_data, Y_data, batchsz, n_fold, seed, half):
    '''
    define data generator by stratified sampling for cross validation
    '''
    np.random.seed(seed)
    
    indice0 = np.where(Y_data == 0)
    indice0 = indice0[0].tolist()
    np.random.shuffle(indice0)
    n0 = int(np.ceil(len(indice0)/5))
    indice1 = np.where(Y_data == 1)
    indice1 = indice1[0].tolist()
    np.random.shuffle(indice1)
    n1 = int(np.ceil(len(indice1)/5))
    
    for i in range(n_fold):
        
        start = n0*i
        end = n0*(i+1)
        if end>len(indice0)-1:
            end = len(indice0)-1
        train_indice = indice0[:start] + indice0[end:]
        test_indice = indice0[start:end]
        x0 = X_data[train_indice]
        y0 = Y_data[train_indice]
        x_test0 = X_data[test_indice]
        y_test0 = Y_data[test_indice]
        
        start = n1*i
        end = n1*(i+1)
        if end>len(indice1)-1:
            end = len(indice1)-1
        train_indice = indice1[:start] + indice1[end:]
        test_indice = indice1[start:end]
        x1 = X_data[train_indice]
        y1 = Y_data[train_indice]
        x_test1 = X_data[test_indice]
        y_test1 = Y_data[test_indice]
        
        x = np.concatenate((x0,x1),axis=0)
        y = np.concatenate((y0,y1),axis=0)
        x_test = np.concatenate((x_test0,x_test1),axis=0)
        y_test = np.concatenate((y_test0,y_test1),axis=0)
        
        if half:
            x,x_test = half_features(x,x_test)
        
        db = tf.data.Dataset.from_tensor_slices((x,y))
        db = db.map(preprocess).shuffle(x.shape[0]).batch(batchsz)

        db_test = tf.data.Dataset.from_tensor_slices((x_test,y_test))
        db_test = db_test.map(preprocess).batch(batchsz)
        yield db, db_test
        
def train(epochs, db, db_test, x_len, optimizer, my_reg, dropout_rate, half):
    '''
    train the model, dispaly useful informatin, and make the plot
    
    epochs: # of epochs
    db: training data
    db_test: test data
    x_len: length of a x vector
    my_reg: regularizer
    dropout_rate: dropout rate
    half: keep half of the feature vector or not
    '''
      
    tf.keras.backend.clear_session()
    # you can use higher-level keras APIs to define and train the model
    # I do it in this way because I've no idea how to use those APIs tocalculate sensitivity and specificity
    model = Sequential([
        layers.Dense(2600, activation=tf.nn.relu,kernel_regularizer=my_reg),
        layers.Dropout(dropout_rate),
        layers.Dense(2048, activation=tf.nn.relu,kernel_regularizer=my_reg),
        layers.Dropout(dropout_rate),
        layers.Dense(1024, activation=tf.nn.relu,kernel_regularizer=my_reg), 
        layers.Dropout(dropout_rate),
        layers.Dense(512, activation=tf.nn.relu,kernel_regularizer=my_reg), 
        layers.Dropout(dropout_rate),
        layers.Dense(256, activation=tf.nn.relu,kernel_regularizer=my_reg),
        layers.Dropout(dropout_rate),
        layers.Dense(128, activation=tf.nn.relu,kernel_regularizer=my_reg),
        layers.Dropout(dropout_rate),
        layers.Dense(64, activation=tf.nn.relu,kernel_regularizer=my_reg),
        layers.Dropout(dropout_rate),
        layers.Dense(32, activation=tf.nn.relu,kernel_regularizer=my_reg),
        layers.Dropout(dropout_rate),
        layers.Dense(2) 
            ])
    
    if half:
        model.build(input_shape=[None, 2*int(np.round(x_len/4.0))])
    else:
        model.build(input_shape=[None, x_len])
    # display model information
    #model.summary()
    
    for epoch in range(epochs):

        for step, (x,y) in enumerate(db):

            with tf.GradientTape() as tape:
                logits = model(x)
                y_onehot = tf.one_hot(y, depth=2)
                # use cross-entropy loss here
                #loss_mse = tf.reduce_mean(tf.losses.MSE(y_onehot, logits))
                loss_ce = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss_ce = tf.reduce_mean(loss_ce)

            grads = tape.gradient(loss_ce, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            #if step % 100 == 0:
            #print(epoch, step, 'loss:', float(loss_ce), float(loss_mse))

        # train statistics
        y_true = tf.convert_to_tensor([],dtype=tf.int32)
        y_pred = tf.convert_to_tensor([],dtype=tf.int32)
        
        for x,y in db:
            logits = model(x)
            # logits => prob, [b, 2]
            prob = tf.nn.softmax(logits, axis=1)
            # [b, 2] => [b], int64
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)
            # append train results
            y_true = tf.concat([y_true,y],axis=0)
            y_pred = tf.concat([y_pred,pred],axis=0)
        # calculate train accuracy, sensitivity, and specificity
        tn,fp,fn,tp = confusion_matrix(y_true.numpy(), y_pred.numpy()).ravel()
        acc_train = (tn+tp)/(tn+tp+fn+fp)
        sen_train = tp/(tp+fn)
        spe_train = tn/(tn+fp)
 
        # test statistics
        y_true = tf.convert_to_tensor([],dtype=tf.int32)
        y_pred = tf.convert_to_tensor([],dtype=tf.int32)
        temp_score = tf.convert_to_tensor([],dtype=tf.float32)
        for x,y in db_test:
            logits = model(x)
            # logits => prob, [b, 2]
            prob = tf.nn.softmax(logits, axis=1)
            temp_score = tf.concat([temp_score,prob[:,1]],axis=0)
            # [b, 2] => [b], int64
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)
            # append test results
            y_true = tf.concat([y_true,y],axis=0)
            y_pred = tf.concat([y_pred,pred],axis=0)
        # calculate test accuracy, sensitivity, and specificity
        tn,fp,fn,tp = confusion_matrix(y_true.numpy(), y_pred.numpy()).ravel()
        acc_test = (tn+tp)/(tn+tp+fn+fp)
        sen_test = tp/(tp+fn)
        spe_test = tn/(tn+fp)
        
        #if epoch%25==0 or epoch == (epochs-1):
        if epoch == (epochs-1):
            print("epoch: ", epoch, 
                  
                  ' train sensi: ', sen_train,
                  ' train speci: ', spe_train,
                  ' train acc: ', acc_train,
                  
                  ' test sensi: ', sen_test,
                  ' test speci: ', spe_test,
                  ' test acc: ', acc_test)
    return sen_test, spe_test, acc_test, y_true.numpy(), temp_score.numpy()

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
import matplotlib.pyplot as plt
def plot_roc_curve(y_true, y_score, title, save_name):
    """plot_roc_curve."""
    fig, ax = plt.subplots()
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    for i in range(6):
        if i<5:
            xs = np.array(y_true[i])
            ys = np.array(y_score[i])
            
            false_positive_rate, true_positive_rate, thresholds = roc_curve(xs, ys)
            interp_tpr = np.interp(mean_fpr, false_positive_rate, true_positive_rate)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            auc_score = roc_auc_score(xs, ys)
            aucs.append(auc_score)
            
            label_name = r'ROC fold %d (AUC = %0.2f )' % (i, auc_score)
            
            ax.plot(false_positive_rate, true_positive_rate, lw=1,label=label_name)
        else:
           
            
            ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            ax.plot(mean_fpr, mean_tpr, color='b',
                    label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                    lw=2, alpha=.8)

            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

            ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
                   title= title)
            ax.legend(loc="lower right")
            plt.savefig(save_name)
            plt.show()
            
def get_names(X_data_name):
    
    if 'AAL' in X_data_name:
        atlas = 'AAL'
    elif 'BASC197' in X_data_name:
        atlas = 'BASC197'
    elif 'BASC444' in X_data_name:
        atlas = 'BASC197'
    elif 'CC200' in X_data_name:
        atlas = 'CC200'
    elif 'CC400' in X_data_name:
        atlas = 'CC400'
    elif 'Dict' in X_data_name:
        atlas = 'Dict'
    elif 'GroupICA' in X_data_name:
        atlas = 'GroupICA'
    elif 'Power' in X_data_name:
        atlas = 'Power'
    
    if 'corr' in X_data_name:
        connectivity = 'Correaltion'
    elif 'part' in X_data_name:
        connectivity = 'Partial Correaltion'
    elif 'tang' in X_data_name:
        connectivity = 'Tangent'
    
    title = 'Model: DNN' + '  Atlas:  ' + atlas + '  Kind: ' + connectivity
    save_name = 'DNN_'+atlas+'_'+connectivity
    
    return title,save_name

def one_setting(np_seed, tf_seed, X_data, Y_data, n_fold,
                batchsz, optimizer, regularizer, dropout_rate,
                half, stratify, title, save_name):
    tf.random.set_seed(tf_seed)
    if stratify:
        cv_data = data_generator_stratify(X_data,Y_data,batchsz,n_fold,np_seed,half)
    else:
        cv_data = data_generator(X_data,Y_data,batchsz,n_fold,np_seed,half)
    cv = 0
    acc_test = 0
    sen_test = 0
    spe_test = 0
    
    y_real = [0,]*n_fold
    y_score = [0,]*n_fold
    x_len = X_data.shape[1]
    for db, db_test in cv_data:
        print("cv: ", cv)
        a,b,c,d,e = train(150,db,db_test,x_len,optimizer,regularizer,dropout_rate,half)
        sen_test += a
        spe_test += b
        acc_test += c
        y_real[cv] = d.tolist()
        y_score[cv] = e.tolist()
        cv += 1
    print("avg test sensitivity ", sen_test/5, "avg test specificity ", spe_test/5, " avg test accuracy ", acc_test/5)
    plot_roc_curve(y_real, y_score, title, save_name)
    
if __name__ == '__main__':
    print('File name: ABIDE_fMRI_DNN.py')