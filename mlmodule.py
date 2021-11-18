from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import joblib
import numpy as np
import logging
logger = logging.getLogger("MLTrack")

#unsupervised learning
def kmeans_clustering(column,top_n_terms,ngram_range=None,fe_type=None,n_clusters=None,max_n_clusters=None):
    """
    K- means clustering for unsupervised learning. User can choose either options:
    (1) provide the number of clusters or
    (2) provide the max number of clusters for kmeans to iterate through, the optimal number of clusters with highest 
    silhouette score will be chosen. Min number of clusters is fixed as 2
    
    params:
    column [series/DataFrame]: column(s) selected for clustering 
                        - series: only one column is selected for clustering (e.g. df["title_clean"])
                        - DataFrame: more than one column is selected for clustering (e.g. df[["title_clean","desc_clean"]])
    top_n_terms[int]: the top n terms in each cluster to be printed out
    ngram_range [tuple(min_n, max_n)]: The lower and upper boundary of the range of n-values for different n-grams to be extracted
                                   - [default] ngram_range of (1, 1) means only unigrams, 
                                   - ngram_range of (1, 2) means unigrams and bigrams, 
                                   - ngram_range of (2, 2) means only bigram
    fe_type[string/None]: Feature extraction type: Choose "bagofwords" for bow or None for default tfidf method
    n_clusters[None/int]: number of clusters. Choose None for option (2)  
    max_n_clusters[None/int]: max number of clusters. Choose None for option (1)  
    """   
    logger.info("kmeans_clustering starts")
        
    silhouette_avg_list = []
    n_clusters_list = []
    dicts = {}
    
    #call feature extraction function    
    ascending = None 
    X = feature_extraction(column,ngram_range,ascending,fe_type)[0]
    X = X.drop(index='sum')
    vec_type = feature_extraction(column,ngram_range,ascending,fe_type)[1]

    #user provides the number of clusters        
    if n_clusters != None:
        model = KMeans(n_clusters = n_clusters, random_state=42)
        model.fit_predict(X)
        labels = model.labels_

        silhouette_score = metrics.silhouette_score(X, labels,random_state=42)
        logger.info("Silhouette score for %s",n_clusters,"clusters is %s",round(silhouette_score,3))
                    
    #user provides the maximum number of clusters 
    if max_n_clusters != None:
        for n_clusters in range(2,max_n_clusters+1): 

            model = KMeans(n_clusters = n_clusters, random_state=42)
            model.fit_predict(X)
            labels = model.labels_

            silhouette_avg = metrics.silhouette_score(X, labels,random_state=42)
            logger.info("For n_clusters = %s", n_clusters,"The silhouette_score is : %s", round(silhouette_avg,3))

            silhouette_avg_list.append(silhouette_avg)
            n_clusters_list.append(n_clusters)

        for i in range(len(n_clusters_list)):
            dicts[n_clusters_list[i]] = silhouette_avg_list[i]

        n_clusters_max = max(dicts,key=dicts.get)
        silhouette_avg_max = max(dicts.values())

        model = KMeans(n_clusters = n_clusters_max, random_state=42)
        model.fit_predict(X)
        labels = model.labels_
        n_clusters = n_clusters_max
        logger.info("\nThe optimal number of clusters selected is %s",n_clusters_max,"with silhouette_score of %s",round(silhouette_avg_max,3),"\n %s") 
        
    logger.info("Top",top_n_terms, %s"terms per cluster:")
    #sort by descending order
    order_centroids = model.cluster_centers_.argsort()[:, ::-1] 
    logger.info("Sorted by descending order")
    terms = vec_type.get_feature_names()
    for i in range(n_clusters):
        logger.info("Cluster %d: %s" % i)
        #top n terms in each cluster
        logger.info(['%s' % terms[ind] for ind in order_centroids[i, :top_n_terms]] %s) 
        logger.info("Shown top n terms in each cluster")
        logger.info("\n %s")
   
    logger.info("kmeans_clustering ends")           
    return labels

def lda(column,n_components,top_n_terms,ngram_range=None):
    """
    LDA for unsupervised learning. Bag of words is selected for feature extraction
    params:
    column [series/DataFrame]: column(s) selected for lda
                        - series: only one column is selected for lda (e.g. df["title_clean"])
                        - DataFrame: more than one column is selected for lda (e.g. df[["title_clean","desc_clean"]])
    n_components[int]: the number of topics/clusters used in the lda_model
    top_n_terms[int]: the top n terms in each topic/cluster to be printed out
    ngram_range [tuple(min_n, max_n)]: The lower and upper boundary of the range of n-values for different n-grams to be extracted
                                   - [default] ngram_range of (1, 1) means only unigrams, 
                                   - ngram_range of (1, 2) means unigrams and bigrams, 
                                   - ngram_range of (2, 2) means only bigram
    
    """
    logger.info("lda starts")
    
    #feature extraction
    ascending = None
    fe_type = "bagofwords"
    vec_type = feature_extraction(column,ngram_range,ascending,fe_type)[1]
    vectorized = feature_extraction(column,ngram_range,ascending,fe_type)[2]

    # Create object for the LDA class 
    lda_model = LatentDirichletAllocation(n_components, random_state = 42)  
    lda_model.fit(vectorized)
    
    # Components_ gives us our topic distribution 
    topic_words = lda_model.components_

    # Top n words for a topic
    for i,topic in enumerate(topic_words):
        logger.info(f"The top {top_n_terms} words for topic #{i} %s")
        logger.info([vec_type.get_feature_names()[index] for index in topic.argsort()[-top_n_terms:]] %s)
        logger.info("\n %s")
    #probabilities of doc belonging to particular topic    
    topic_results = lda_model.transform(vectorized) 
    logger.info("Shown probabilities of doc belonging to particular topic ")
    
    logger.info("lda ends")   
    
    return topic_results.argmax(axis=1)

def nmf(column,n_components,top_n_terms,fe_type,ngram_range=None):
    """
    Non-negative matrix factorization for unsupervised learning.
    params:
    column [series/DataFrame]: column(s) selected for NMF 
                        - series: only one column is selected for NMF (e.g. df["title_clean"])
                        - DataFrame: more than one column is selected for NMF (e.g. df[["title_clean","desc_clean"]])
    n_components[int]: the number of topics/clusters used in NMF
    top_n_terms[int]: the top n terms in each topic/cluster to be printed out
    fe_type[string/None]: Feature extraction type: Choose "bagofwords" for bow or None for default tfidf method
    ngram_range [tuple(min_n, max_n)]: The lower and upper boundary of the range of n-values for different n-grams to be extracted
                                   - [default] ngram_range of (1, 1) means only unigrams, 
                                   - ngram_range of (1, 2) means unigrams and bigrams, 
                                   - ngram_range of (2, 2) means only bigram
    """
    logger.info("nmf starts")    
    
    #feature extraction
    ngram_range = None
    ascending = None
    vec_type = feature_extraction(column,ngram_range,ascending,fe_type)[1]
    vectorized = feature_extraction(column,ngram_range,ascending,fe_type)[2]

    # Create object for the NMF class 
    nmf_model = NMF(n_components,random_state=42)
    nmf_model.fit(vectorized)
    
    # Components_ gives us our topic distribution 
    topic_words = nmf_model.components_

    # Top n words for a topic

    for i,topic in enumerate(topic_words):
        logger.info(f"The top {top_n_terms} words for topic #{i} %s")
        logger.info([vec_type.get_feature_names()[index] for index in topic.argsort()[-top_n_terms:]] %s)
        logger.info("\n %s")
        
    topic_results = nmf_model.transform(vectorized) 
    
    logger.info("nmf ends") 
    
    return topic_results.argmax(axis=1)

#supervised learning

#Machine Learning
def supervised_lng(df,user_outpath,target,test_size,ngram_range=None,fe_type=None,model_type=None,ascend=None):

    """
    Consists of 3 supervised machine learning methods: RandomForest (Default), Naive Bayes(optional, SVM (optional)
    
    X[series/DataFrame]: column(s) of text for supervised learning
                        - series: only one column is selected (e.g. df["title_clean"])
                        - DataFrame: more than one column is selected(e.g. df[["title_clean","desc_clean"]])
    y[series]: target 
    test_size[float/int]: If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. 
                          If int, represents the absolute number of test samples.
    ngram_range [tuple(min_n, max_n)]: The lower and upper boundary of the range of n-values for different n-grams to be extracted
                                       -[DEFAULT] ngram_range of (1, 1) means only unigrams, 
                                       - ngram_range of (1, 2) means unigrams and bigrams, 
                                       - ngram_range of (2, 2) means only bigram
    fe_type[None/string]: Feature extraction type: Choose "bagofwords" or None for default tfidf method
    model_type[None/string]: Choose ML algorithm 
                            - None (Default algorithm is Random Forest)
                            - 'NB'(To choose Naive Bayes as ML algorithm), 
                            - 'SVM'(To choose Support Vector Machine as ML algorithm)
    ascend[True/False/None]:  - None (Default: Confusion matrix is arranged in alphabetical order)
                              - True(Confusion matrix arranged in ascending order of accuracy % per label), 
                              - False(Confusion matrix arranged in descending order of accuracy % per label)  
    save_path[None/string]: Path to save model
                            - None (Default - Model is not saved)
                            - String (Model is saved as model.joblib in the save_path specified as a string)
        
    """
    logger.info("supervised_lng starts")    
    logger.info("Target distribution: %s",df[target].value_counts())
    X= df.drop([target],axis=1)
    y= df[target]   
    
    #TRAIN-TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 42)
    logger.info("Train-test split completed with %s",(1-test_size)*100,"- %s",test_size*100,"split in train-test %s")
    logger.info("Shape of X_train is: %s", X_train.shape)
    logger.info("Shape of X_test is: %s",X_test.shape)
    logger.info("Shape of y_train is: %s",y_train.shape)
    logger.info("Shape of y_test is: %s",y_test.shape)
    #concat the columns into one string if there is more than one column 
    if type(X_train) == pd.DataFrame:
    logger.info("columns concatted into one string if there is more than one column")
        X_train = X_train.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)         
    #concat the columns into one string if there is more than one column    
    if type(X_test) == pd.DataFrame:
    logger.info("columns concatted into one string if there is more than one column")
        X_test = X_test.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

    
    #FEATURE EXTRACTION
    column = X_train       
    ascending = None
    #fit_transform X_train
    X_train = feature_extraction(column,ngram_range,ascending,fe_type)[2]
    #only transform X_test
    vec_type = feature_extraction(column,ngram_range,ascending,fe_type)[1]
    X_test = vec_type.transform(X_test)
    
    
    logger.info("Shape of X_train after feature extraction: %s",X_train.shape)
    logger.info("Shape of X_test after feature extraction: %s",X_test.shape)
    
    #MODEL BUILDING
    if model_type == None:
        #random forest is chosen by default
        model = RandomForestClassifier(random_state = 42)
    
    if model_type == "NB":
        model = MultinomialNB()
                   
    if model_type == "SVM":
        model = svm.SVC(random_state = 42)
    
    model.fit(X_train, y_train) 
    
    #MODEL SAVING
    
    joblib.dump(model, user_outpath + "model.joblib")
    logger.info("Model saved! %s")

    # predicting test set results
    y_pred = model.predict(X_test)

    # MODEL EVALUATION  
   
    #print("Overall accuracy achieved is" + str(round(metrics.accuracy_score(y_test, y_pred)*100,2)) + "%")
    logger.info("Classification report:\n %s",metrics.classification_report(y_test, y_pred,zero_division=0))
    
    #overall accuracy
    overall_acc = round(metrics.accuracy_score(y_test, y_pred)*100,2)
    overall_acc = {'Overall Acc %':overall_acc}
    overall_acc = pd.DataFrame([overall_acc])
    overall_acc.to_csv(user_outpath+"Overall_Accuracy.csv")

    #classification report
    report = metrics.classification_report(y_test, y_pred,zero_division=0,output_dict=True)
    report = pd.DataFrame(report).transpose()
    report.to_csv(user_outpath+"Classification_Report.csv")

    #confusion matrix with accuracies for each label
    class_accuracies = []

    for class_ in y_test.sort_values(ascending= True).unique():
        class_acc = round(np.mean(y_pred[y_test == class_] == class_)*100,2)
        class_accuracies.append(class_acc)
    class_acc = pd.DataFrame(class_accuracies,index=y_test.sort_values(ascending= True).unique(),columns= ["Accuracy %"])

    cf_matrix = pd.DataFrame(
        metrics.confusion_matrix(y_test, y_pred, labels= y_test.sort_values(ascending= True).unique()), 
        index=y_test.sort_values(ascending= True).unique(), 
        columns=y_test.sort_values(ascending= True).unique()
    )
    
    if ascend == None:
        cf_matrix = pd.concat([cf_matrix,class_acc],axis=1)
    else:
        cf_matrix = pd.concat([cf_matrix,class_acc],axis=1).sort_values(by=['Accuracy %'], ascending=ascend)
    logger.info("supervised_lng ends")     
     
    cf_matrix.to_csv(user_outpath+"Confusion_Matrix.csv",index=False)  

#Deep Learning 
def deep_lng(df,user_outpath,target,test_size,ngram_range,fe_type,hidden_layer_sizes=None,activation=None,solver=None,learning_rate=None,max_iter=None,ascend=None):
    """
     Deep learning method: MultiLayer Perceptron

    X[series/DataFrame]: column(s) of text for deep learning
                        - series: only one column is selected (e.g. df["title_clean"])
                        - DataFrame: more than one column is selected(e.g. df[["title_clean","desc_clean"]])   
    y[series]: target
    test_size[float/int]: If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. 
                          If int, represents the absolute number of test samples.
    ngram_range [tuple(min_n, max_n)]: The lower and upper boundary of the range of n-values for different n-grams to be extracted
                                       - ngram_range of (1, 1) means only unigrams, 
                                       - ngram_range of (1, 2) means unigrams and bigrams, 
                                       - ngram_range of (2, 2) means only bigram
    fe_type[string]: Feature extraction type: Choose "bagofwords" or "tfidf" method
    hidden_layer_sizes[tuple],default = (100): To set the number of layers and the number of nodes.
                                               Each element in the tuple represents the number of nodes,
                                               length of tuple denotes the total number of hidden layers in the network
    activation["identity", "logistic", "tanh","relu"], default="relu": Activation function for the hidden layer.
    solver["lbfgs", "sgd", "adam"], default="adam": The solver for weight optimization.
    learning_rate["constant", "invscaling", "adaptive"], default="constant": Learning rate schedule for weight updates
    max_iter[int], default=200: Maximum number of iterations. The solver iterates until convergence or this number of iterations.
    ascend [True/False/None]: - None (Default: Confusion matrix is arranged in alphabetical order)
                                 - True(Confusion matrix arranged in ascending order of accuracy % per label), 
                                 - False(Confusion matrix arranged in descending order of accuracy % per label)                            
    save_path[None/string]: Path to save model
                            - None (Default - Model is not saved)
                            - String (Model is saved as model.joblib in the save_path specified as a string)    
    """
    logger.info("deep_lng starts")  
    logger.info("Target distribution:",df[target].value_counts())
    X= df.drop([target],axis=1)
    y= df[target]   
    
    #train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 42)
    logger.info("Train-test split completed with %s",(1-test_size)*100,"- %s",test_size*100,"split in train-test %s")
    logger.info("Shape of X_train is: %s", X_train.shape)
    logger.info("Shape of X_test is: %s",X_test.shape)
    logger.info("Shape of y_train is: %s",y_train.shape)
    logger.info("Shape of y_test is: %s",y_test.shape)
    #concat the columns into one string if there is more than one column 
    if type(X_train) == pd.DataFrame: 
        X_train = X_train.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)         
    #concat the columns into one string if there is more than one column     
    if type(X_test) == pd.DataFrame: 
        X_test = X_test.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        
    #FEATURE EXTRACTION
    column = X_train
    ascending = None
    #fit_transform X_train
    X_train = feature_extraction(column,ngram_range,ascending,fe_type)[2]
    #only transform X_test
    vec_type = feature_extraction(column,ngram_range,ascending,fe_type)[1]
    X_test = vec_type.transform(X_test)
    logger.info("Shape of X_train after feature extraction: %s",X_train.shape)
    logger.info("Shape of X_test after feature extraction: %s",X_test.shape)
    
    #MODEL BUILDING
    #default hypermarameters
    if hidden_layer_sizes == None:
        hidden_layer_sizes = (100)
    if activation == None:
        activation = "relu"
    if solver == None:
        solver = "adam"
    if learning_rate == None:
        learning_rate = "constant"
    if max_iter == None:
        max_iter = 200
    
    logger.info("Hidden layer sizes: %s", hidden_layer_sizes,", Activation: %s",activation,", Solver: %s",solver,", Learning rate: %s",learning_rate,", Max iteration: %s",max_iter)
    
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, max_iter=max_iter,verbose = False,random_state=42)
    model.fit(X_train,y_train)
    
    
    #MODEL SAVING    
    joblib.dump(model, user_outpath + "mlpmodel.joblib")
    logger.info("Model saved! %s")

    # predicting test set results
    y_pred = model.predict(X_test)

    # MODEL EVALUATION  
   
    #print('Overall accuracy achieved is ' + str(round(metrics.accuracy_score(y_test, y_pred)*100,2)) + "%")
    logger.info("Classification report:\n %s",metrics.classification_report(y_test, y_pred,zero_division=0))
    
    #overall accuracy
    overall_acc = round(metrics.accuracy_score(y_test, y_pred)*100,2)
    overall_acc = {'Overall Acc %':overall_acc}
    overall_acc = pd.DataFrame([overall_acc])
    overall_acc.to_csv(user_outpath+"Overall_Accuracy.csv")

    #classification report
    report = metrics.classification_report(y_test, y_pred,zero_division=0,output_dict=True)
    report = pd.DataFrame(report).transpose()
    report.to_csv(user_outpath+"Classification_Report.csv")

    #confusion matrix with accuracies for each label
    class_accuracies = []

    for class_ in y_test.sort_values(ascending= True).unique():
        class_acc = round(np.mean(y_pred[y_test == class_] == class_)*100,2)
        class_accuracies.append(class_acc)
    class_acc = pd.DataFrame(class_accuracies,index=y_test.sort_values(ascending= True).unique(),columns= ["Accuracy %"])

    cf_matrix = pd.DataFrame(
        metrics.confusion_matrix(y_test, y_pred, labels= y_test.sort_values(ascending= True).unique()), 
        index=y_test.sort_values(ascending= True).unique(), 
        columns=y_test.sort_values(ascending= True).unique()
    )
    
    if ascend == None:
        cf_matrix = pd.concat([cf_matrix,class_acc],axis=1)
    else:
        cf_matrix = pd.concat([cf_matrix,class_acc],axis=1).sort_values(by=['Accuracy %'], ascending=ascend)
    logger.info("deep_lng ends")          
    
    cf_matrix.to_csv(user_outpath+"Confusion_Matrix.csv",index=False)   

#similarity metrics

#Cosine Similarity 

def cosinesimilarity(column,user_outpath,threshold=None,total_rows = None,base_row=None,ngram_range=None,fe_type=None,ascending=None):
    """
    Compute the cosine similarity between rows of texts. User can 
    a) fix number of rows for comparison, each row will be taken as base and compared with the rest
    b) fix one row as base, comparison will be done with all the other rows
    
    params:    
    column[series/DataFrame]: column(s) of text for row wise similarity comparison
                        - series: only one column is selected (e.g. df["title_clean"])
                        - DataFrame: more than one column is selected(e.g. df[["title_clean","desc_clean"]])  
    threshold[None/float]: cut off value for the cosine similarity, only texts with values above or equal to threshold
                           will be printed
                        - None: Default threhold is 0.5
                        - float: any value between 0 and 1 
    total_rows[None/int]: Number of rows for comparison, choose None for option b 
    base_row[None/int]: Row fixed as base, choose None for option a 
    ngram_range [tuple(min_n, max_n)]: The lower and upper boundary of the range of n-values for different n-grams to be extracted
                                       - ngram_range of (1, 1) means only unigrams, 
                                       - ngram_range of (1, 2) means unigrams and bigrams, 
                                       - ngram_range of (2, 2) means only bigram
    fe_type[None/string]: Feature extraction type: Choose "bagofwords" or None for tfidf
    ascending [True/False/None]: - [default] None (words arranged in alphabetical order)
                                 - True(words arranged in ascending order of sum), 
                                 - False(words arranged in descending order of sum)  
    
    """
    logger.info("cosinesimilarity starts")
    #concat the columns into one string if there is more than one column
    if type(column) == pd.DataFrame:  
        column = column.apply(lambda row: ' '.join(row.values.astype(str)), axis=1) 
                
    #feature extraction              
    X = feature_extraction(column=column,ngram_range=ngram_range,ascending=None,fe_type=fe_type)[0]
    X = X.drop(["sum"],axis = 0)
    
    #Get cosine similarity matrix
    similarity_matrix = pd.DataFrame(cosine_similarity(X))
    
    #threshold
    if threshold == None:
        threshold = 0.5
        
    #fix number of rows for comparison, each row will be taken as base and compared with the rest   
    if total_rows !=None: 
        logger.info("Fix number of rows for comparison, each row will be taken as base and compared with the rest %s")
        results_append = []
        for base in range(total_rows):            
            #Create empty df
            column_names = ["Base Index","Index", "Similarity Score", "Text"]
            results = pd.DataFrame(columns = column_names)
            
            #compare base with other index
            for i in range(total_rows): 
                
                if similarity_matrix.iloc[base,i] >= threshold:
                logger.info("Comparison shown if silarity metric is more than threshold %s")    
                    new_row = {'Base Index':base,'Index':i,'Similarity Score':round(similarity_matrix.iloc[base,i],4), 'Text':column.iloc[i]}
                    #append row to the dataframe
                    results = results.append(new_row, ignore_index=True)
                
                    if ascending != None:            
                        results = results.sort_values(by ='Similarity Score', axis = 0,ascending=ascending)
                        
            #display(results)
            results_append.append(results)
        results_append = pd.concat(results_append)
           #display(results_append)         
        results_append.to_csv(user_outpath+"Cosine_Similarity.csv",index=False)
    
    #fix base_row index for comparison with all indexes        
    if base_row !=None: 
        logger.info ("Fix one row as base, comparison will be done with all the other rows %s")          
        #Create empty df
        column_names = ["Base Index","Index", "Similarity Score", "Text"]
        results = pd.DataFrame(columns = column_names)
        
        #compare base_row with other index
        for i in range(len(column)): 
            #print if comparison shows that silarity metric is more than threshold
            if similarity_matrix.iloc[base_row,i] >= threshold: 
                new_row = {'Base Index':base_row,'Index':i, 'Similarity Score':round(similarity_matrix.iloc[base_row,i],4), 'Text':column.iloc[i]}
                #append row to the dataframe
                results = results.append(new_row, ignore_index=True)
                if ascending != None:            
                    results = results.sort_values(by ='Similarity Score', axis = 0,ascending=ascending)  
        #display(results)
    logger.info("cosinesimilarity ends")        
        
        results.to_csv(user_outpath+"Cosine_Similarity.csv",index=False)  

#Jaccard Similarity 

def jaccardsimilarity(column,user_outpath,threshold=None,total_rows = None,base_row=None,ascending=None):
    """
    Compute the jaccard similarity between texts. User can 
    a) fix number of rows for comparison, each row will be taken as base and compared with the rest
    b) fix one row as base, comparison will be done with all the other rows
    
    params:
    column[series/DataFrame]: column(s) of text for row wise similarity comparison
                        - series: only one column is selected (e.g. df["title_clean"])
                        - DataFrame: more than one column is selected(e.g. df[["title_clean","desc_clean"]]) 
    threshold[None/float]: cut off value for the jaccard similarity, only texts with values above or equal to threshold
                           will be printed
                        - None: Default threhold is 0.5
                        - float: any value between 0 and 1 
    total_rows[None/int]: Number of rows for comparison, choose None for option b 
    base_row[None/int]: Row fixed as base, choose None for option a 
    ascending [True/False/None]: - [default] None (words arranged in alphabetical order)
                                 - True(words arranged in ascending order of sum), 
                                 - False(words arranged in descending order of sum)  
    
    """     
    logger.info("jaccardsimilarity starts")
            
    #jaccard score computation
    def get_jaccard_sim(str1, str2):        
        a = set(str1.split()) 
        b = set(str2.split())
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))
    
    #concat the columns into one string if there is more than one column 
    if type(column) == pd.DataFrame: 
        column = column.apply(lambda row: ' '.join(row.values.astype(str)), axis=1) 
       
    #threshold
    if threshold == None:
        threshold = 0.5
        
    #fix number of rows for dcomparison, each row will be taken as base and compared with the rest
    if total_rows !=None: 
        logger.info("Fix number of rows for comparison, each row will be taken as base and compared with the rest %s")
        results_append = []
        for base in range(total_rows):
            
            #Create empty df
            column_names = ["Base Index","Index", "Similarity Score", "Text"]
            results = pd.DataFrame(columns = column_names)                   
            
            #compare base with other index
            for i in range(total_rows): 
                jac_score =  round(get_jaccard_sim(column.iloc[base],column.iloc[i]),4)
                logger.info(Comparison shown if similarity metric is more than threshold %s")
                if jac_score >= threshold: 
                    new_row = {"Base Index":base,'Index':i, 'Similarity Score':jac_score, 'Text':column.iloc[i]}
                    #append row to the dataframe
                    results = results.append(new_row, ignore_index=True)
                
                if ascending != None:            
                    results = results.sort_values(by ='Similarity Score', axis = 0,ascending=ascending)  
                    
            #display(results)
            results_append.append(results)
        results_append = pd.concat(results_append)
        #display(results_append)         
        results_append.to_csv(user_outpath+"Jaccard_Similarity.csv",index=False)
        
    if base_row != None: #fix base_row index for comparison with all indexes
       
        logger.info ("Fix one row as base, comparison will be done with all the other rows %s") 
        #Create empty df
        column_names = ["Base Index","Index", "Similarity Score", "Text"]
        results = pd.DataFrame(columns = column_names)                   
        
        #compare base_row with other index    
        for i in range(len(column)): 
            jac_score = round(get_jaccard_sim(column.iloc[base_row],column.iloc[i]),4)
            #print if comparison shows that silarity metric is more than threshold
            if jac_score >= threshold: 
                new_row = {"Base Index":base_row,'Index':i, 'Similarity Score':jac_score, 'Text':column.iloc[i]}
                #append row to the dataframe
                results = results.append(new_row, ignore_index=True)
            if ascending != None:            
                results = results.sort_values(by ='Similarity Score', axis = 0,ascending=ascending)  
        #display(results)
    logger.info("jaccardsimilarity ends")
    
        results.to_csv(user_outpath+"Jaccard_Similarity.csv",index=False)  

