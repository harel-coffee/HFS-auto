def fetsel(X,y,n):                           # X is the attributes ,y is the class , n specifies the number of class labels
    #Importing Libraries
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix
    from sklearn.externals import joblib
    from sklearn import metrics
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics.cluster import normalized_mutual_info_score
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.model_selection import train_test_split
    from sklearn.feature_selection import RFE
    import time
    import warnings; warnings.simplefilter('ignore')

    #creating a list of features name
    fet=[]
    for i in X.columns:
        fet.append(i)
    #Normalizing
    X=MinMaxScaler().fit_transform(X) # or Standard scaler based on the dataset
    x=pd.DataFrame(X)
    #splitting
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=9)
    rf = RandomForestClassifier(random_state=0, n_jobs=-1) #or (n_estimators = 100, criterion = 'entropy', random_state = 42, n_jobs = -1)
    x_train.columns=fet
    x_test.columns=fet
    a=x.values
    b=y.values
    b=b.flatten()
    S=[]
    for h in range(a.shape[1]):
        kmeans =  MiniBatchKMeans(init='k-means++', n_clusters=np.unique(b).shape[0],batch_size=6)
        ff=kmeans.fit_predict(a[:,h].reshape(-1,1))   # clustering each feature
        s=normalized_mutual_info_score(b,ff)          # cluster quality
        S.append(s)                                   # adding into the list
    df = pd.DataFrame(list(zip(S, fet)),columns =['val', 'name'])  # creating a dataframe of the rankings
    dfObj = df.sort_values(by ='val',ascending=False )          # rakings in descending order
    dfObj=dfObj.reset_index(drop=True)
    # FEATURE SELECTION
    lst=[dfObj['name'][0]]                                       #initial list containing first ranked feature
    prev=0
    for i in range(0,len(dfObj)):
        x_train1=x_train[lst]
        x_test1=x_test[lst]
        rf.fit(x_train1, y_train.values.ravel())                #fitting the model with the selected features
        y_pred = rf.predict(x_test1)
        acc=accuracy_score(y_test, y_pred)
        print(lst)
        print("ACCURACY: "+str(acc))
        if acc >prev:                                           #comparing the accuracy with the previous subset
            if(i!=len(dfObj)-1):
                lst.append(dfObj['name'][i+1])
                prev=acc
            else:
                print(lst)
        else:
            del lst[-1]
            if(i!=len(dfObj)-1):
                lst.append(dfObj['name'][i+1])
            else:
                print(lst)
    tf=len(lst)

    #Result Section
    
    print("CONSIDERING ALL FEATURES")
    rf1=rf.fit(x_train, y_train)
    y_pred = rf1.predict(x_test)                                     #model fitting with all the features
    acc=accuracy_score(y_test, y_pred)
    print("ACCURACY:   "+str(acc))
    if n==2:
        acu=roc_auc_score(y_test,y_pred)
        print("AUC:   "+ str(acu))
    print("\nClassification - \n")
    print(classification_report(y_test,y_pred, digits=4))

    
    print("RESULT FOR KNMIFI")
    rf1=rf.fit(x_train[lst], y_train)
    y_pred = rf1.predict(x_test[lst])                               #model fitting with the selected features KNMIFI
    acc=accuracy_score(y_test, y_pred)
    print("Accuracy :   "+str(acc))
    if n==2:
        acu=roc_auc_score(y_test,y_pred)
        print("AUC :"+str(acu))
    print("\nClassification - \n")
    print(classification_report(y_test,y_pred, digits=4))
    
    print("For KNMILFE")
    dfObjTemp = df.sort_values(by ='val',ascending=True)
    dfObjTemp=dfObjTemp.reset_index(drop=True)
    lst=[]
    for i in range(0,len(dfObjTemp)):
        lst.append(dfObjTemp['name'][i])    # adding all the feautres in the initial list
    acc=0
    prev=0
    for i in range(0,len(dfObjTemp)):
        x_train1=x_train[lst]
        x_test1=x_test[lst]
        x_train1=pd.DataFrame(x_train1)
        x_test1=pd.DataFrame(x_test1)
        rf.fit(x_train1, y_train.values.ravel())
        y_pred = rf.predict(x_test1)
        acc=accuracy_score(y_test, y_pred)
        print("Removing " +str(i)+"   "+str(acc))
        if acc > prev:
            prev=acc                                                   #selecting the best accuracy
            j=i
        del lst[0]                                                     #removing the least ranked features
    lst1=[]
    for i in range(0,len(dfObjTemp)):
        lst1.append(dfObjTemp['name'][i])
    lst1=lst1[j:]
    rf1=rf.fit(x_train[lst1], y_train)                               
    y_pred = rf1.predict(x_test[lst1])                                   #fitting the model with the least ranked features eliminated
    acc=accuracy_score(y_test, y_pred)
    print("Accuracy: "+str(acc))
    if n==2:
        acu=roc_auc_score(y_test,y_pred)
        print("AUC:   "+acu) 
    print("\nClassification - \n")
    print(classification_report(y_test,y_pred, digits=4)) 
