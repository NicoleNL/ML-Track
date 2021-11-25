# import UsrIntel.R1
import json     
import pandas as pd
import logging  
import argparse
from dataloading import data_loading
from datapreprocessing import df_manipulation,word_contractions,lowercase,remove_htmltag_url,remove_irrchar_punc,remove_num,remove_multwhitespace,remove_stopwords,remove_freqwords,remove_rarewords
from datapreprocessing import custom_taxo,stem_words,lemmatize_words,feature_extraction
from mlmodule import kmeans_clustering,lda,nmf,supervised_lng,deep_lng,cosinesimilarity,jaccardsimilarity

#read config file and call the other functions
def main(config_file):         
    
    with open(config_file) as f:
        data = json.load(f)
        
    user_outpath = data["user_outpath"]    
    log_path = data["log_path"]
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p', filename=log_path+"logs.log", filemode='w')
    logger = logging.getLogger("MLTrack")      
    
    
    email = data["Email"]
    logger.info("########-----------------------------------------############")
    logger.info("USER'S EMAIL: %s",email)
    logger.info("########-----------------------------------------############")
    
        
    #---------DATA LOADING----------#
    logger.info("Data Loading starts")
    dl = data["DataLoading"]
    path,start_date,stop_date = dl['path'],dl['start_date'],dl['stop_date']  
    df = data_loading(path=path,start_date=start_date,stop_date=stop_date)
    logger.info("Data Loading ends")
        
    #---------DATA PREPROCESSING----------# 
    logger.info("Data Preprocessing starts")
    #df_manipulation
    dm = data["DataPreprocessing"]["df_manipulation"]
    how,col_selection,keep,subset = dm['how'],dm['col_selection'],dm['keep'],dm['subset']  
    df = df_manipulation(df,how=how,col_selection=col_selection,keep=keep,subset=subset)
   
    df_all = df.copy() #id, raw text -> data preprocessing final file
    df_out = df.copy() #id, raw text -> ml output
    df = df.drop("id",axis=1) #raw text for data preprocessing
    
    #remove target from df for supervised    
    if data["DataPreprocessing"]["target"]["enable"]:
        target = data["DataPreprocessing"]["target"]["column"]
        target = df[[target]] 
        df = df.drop(target,axis=1)
        df_all = df_all.drop(target,axis=1)
              
#    word_contractions
    wordcont = data["DataPreprocessing"]["word_contractions"]["enable"]    
    if wordcont:
        df = word_contractions(df)
        df_all = pd.concat([df_all,df],axis=1) #raw text, raw text after contractions
            
    #lowercase
    lower = data["DataPreprocessing"]["lowercase"]["enable"]      
    if lower:
        df = lowercase(df)
        df_all = pd.concat([df_all,df],axis=1)

            
   #Remove html tag and url
    tagrem = data["DataPreprocessing"]["remove_htmltag_url"]["enable"] 
    if tagrem:
        df = remove_htmltag_url(df)
        df_all = pd.concat([df_all,df],axis=1)

    #Remove irrelevant characters and punctuation
    pc = data["DataPreprocessing"]["remove_irrchar_punc"]
    puncrem,char = pc["enable"],pc["char"] 
    if puncrem:
        df = remove_irrchar_punc(df,char=char)
        df_all = pd.concat([df_all,df],axis=1)
        
    #Remove numbers 
    numrem = data["DataPreprocessing"]["remove_num"]["enable"]
    if numrem:
        df = remove_num(df)
        df_all = pd.concat([df_all,df],axis=1)

    # Remove multiple whitespace
    wsrem = data["DataPreprocessing"]["remove_multwhitespace"]["enable"]
    if wsrem:
        df = remove_multwhitespace(df)
        df_all = pd.concat([df_all,df],axis=1)

    # Remove stopwords
    sw = data["DataPreprocessing"]["remove_stopwords"]
    stoprem,extra_sw,remove_sw = sw["enable"],sw["extra_sw"],sw["remove_sw"]
    if stoprem:
        df = remove_stopwords(df,extra_sw=extra_sw,remove_sw=remove_sw)
        df_all = pd.concat([df_all,df],axis=1)
        
    # Remove frequent words
    fw = data["DataPreprocessing"]["remove_freqwords"]
    freqrem,n = fw["enable"],fw["n"]
    if freqrem:
        df = remove_freqwords(df,n)
        df_all = pd.concat([df_all,df],axis=1)
    
    # Remove rare words
    rw = data["DataPreprocessing"]["remove_rarewords"]
    rarerem,n = rw["enable"],rw["n"]    
    if rarerem:
        df = remove_rarewords(df,n)
        df_all = pd.concat([df_all,df],axis=1)
        
    # Custom taxonomy
    ct = data["DataPreprocessing"]["custom_taxo"]
    taxo,remove_taxo,include_taxo = ct["enable"], ct["remove_taxo"], ct["include_taxo"]
    if taxo:
        df = custom_taxo(df,remove_taxo,include_taxo)
        df_all = pd.concat([df_all,df],axis=1)     
        
    # Stemming
    st = data["DataPreprocessing"]["stem_words"]
    stem,stemmer_type = st["enable"],st["stemmer_type"]     
    if stem:
        df = stem_words(df,stemmer_type)
        df_all = pd.concat([df_all,df],axis=1)            
    
    #Lemmatization
    lem = data["DataPreprocessing"]["lemmatize_words"]
    lemma,lemma_type = lem["enable"], lem["lemma_type"]      
    if lemma:
        df = lemmatize_words(df,lemma_type)
        df_all = pd.concat([df_all,df],axis=1)
    
    #column bind target to df
    if data["DataPreprocessing"]["target"]["enable"]:
        df = pd.concat([target,df],axis=1)
        df_all = pd.concat([target,df_all],axis=1) 
      
    df_all.to_csv(user_outpath+"preprocessed_text.csv",index=False) #save after data preprocessing   
    logger.info("Data Preprocessing ends")
    
    #---------------ML module---------------# 
    logger.info("Ml module starts")
    
    elk_outpath = data["UnsupervisedLearning"]["elk_outpath"]
    
    ####---Unsupervised Learning---###
    #check output save in user or ELK folder
    if data["UnsupervisedLearning"]["Output"]["User"]:#store output in user folder 
        outpath = user_outpath
    if data["UnsupervisedLearning"]["Output"]["ELK"]: #store output in ELK folder
        outpath = elk_outpath 

    #k-means clustering 
    km= data["UnsupervisedLearning"]["kmeans_clustering"]
    kmeans = km["enable"]
    if kmeans:
        top_n_terms,ngram_range,fe_type,n_clusters,max_n_clusters= km["top_n_terms"],km["ngram_range"],km["fe_type"],km["n_clusters"],km["max_n_clusters"]        
        df_out["cluster"]=kmeans_clustering(column=df,outpath=outpath,top_n_terms=top_n_terms,ngram_range=ngram_range,fe_type=fe_type,n_clusters=n_clusters,max_n_clusters=max_n_clusters)
        df_out.to_csv(outpath+"KMeansClustering_output.csv",index=False)          
        logger.info("K-means clustering results saved in %s as KMeansClustering_output.csv",outpath) 
        
    #LDA
    lda_m= data["UnsupervisedLearning"]["lda"]
    LatentDirichletAllocation = lda_m["enable"]
    if LatentDirichletAllocation:
        n_components,top_n_terms,ngram_range= lda_m["n_components"],lda_m["top_n_terms"],lda_m["ngram_range"]       
        df_out["cluster"]=lda(column=df,outpath=outpath,n_components=n_components,top_n_terms=top_n_terms,ngram_range=ngram_range)            
        df_out.to_csv(outpath+"LatentDirichletAllocation_output.csv",index=False)       
        logger.info("LDA results saved in %s as LatentDirichletAllocation_output.csv",outpath) 

    #NMF Factorization
    nmf_m= data["UnsupervisedLearning"]["nmf"]
    NonNegativeMatrixFactorization = nmf_m["enable"]
    if NonNegativeMatrixFactorization:
        n_components,top_n_terms,fe_type,ngram_range= nmf_m["n_components"],nmf_m["top_n_terms"],nmf_m["fe_type"],nmf_m["ngram_range"]
        df_out["cluster"]=nmf(column=df,outpath=outpath,n_components=n_components,top_n_terms=top_n_terms,fe_type=fe_type,ngram_range=ngram_range)              
        df_out.to_csv(outpath+"NonNegativeMatrixFactorization_output.csv",index=False) 
        logger.info("NMF results saved in %s as NonNegativeMatrixFactorization_output.csv",outpath) 
            
    
    #####---Similarity Metrics-----####
    #Cosine Similarity
    cs= data["SimilarityMetrics"]["cosinesimilarity"]
    cosinesim = cs["enable"]
    if cosinesim:
        threshold,total_rows,base_row,ngram_range,fe_type,ascending= cs["threshold"],cs["total_rows"],cs["base_row"],cs["ngram_range"],cs["fe_type"],cs["ascending"]        
        cosinesimilarity(column=df,user_outpath=user_outpath,threshold=threshold,total_rows=total_rows,base_row=base_row,ngram_range=ngram_range,fe_type=fe_type,ascending=ascending)

    #Jaccard Similarity
    js= data["SimilarityMetrics"]["jaccardsimilarity"]
    jaccardsim = js["enable"]
    if jaccardsim:
        threshold,total_rows,base_row,ascending= js["threshold"],js["total_rows"],js["base_row"],js["ascending"]
        jaccardsimilarity(column=df,user_outpath=user_outpath,threshold=threshold,total_rows=total_rows,base_row=base_row,ascending=ascending)

    #####-----Supervised Learning----####
    
         
    #Machine Learning
    sl= data["SupervisedLearning"]["supervised_lng"]
    SupervisedLearning = sl["enable"]
    if SupervisedLearning:
        target,test_size,ngram_range,fe_type,model_type,ascend= sl["target"],sl["test_size"],sl["ngram_range"],sl["fe_type"],sl["model_type"],sl["ascend"]
        supervised_lng(df=df,user_outpath=user_outpath,target=target,test_size=test_size,ngram_range=ngram_range,fe_type=fe_type,model_type=model_type,ascend=ascend)

    #Deep Learning
    dl= data["SupervisedLearning"]["deep_lng"]
    DeepLearning = dl["enable"]
    if DeepLearning:
        target,test_size,ngram_range,fe_type,hidden_layer_sizes,activation,solver,learning_rate,max_iter,ascend= dl["target"],dl["test_size"],dl["ngram_range"],dl["fe_type"],dl["hidden_layer_sizes"],dl["activation"],dl["solver"],dl["learning_rate"],dl["max_iter"],dl["ascend"]
        deep_lng(df=df,user_outpath=user_outpath,target=target,test_size=test_size,ngram_range=ngram_range,fe_type=fe_type,hidden_layer_sizes=hidden_layer_sizes,activation=activation,solver=solver,learning_rate=learning_rate,max_iter=max_iter,ascend=ascend)
    
    logger.info("Ml module ends")

    
if __name__ == '__main__':  
    # config_file = 'C:/Users/nchong/OneDrive - Intel Corporation/Documents/Debug Similarity Analytics and Bucketization Framework/ML_Testing/config.json'    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',type=str, required=True)
    args = parser.parse_args()
    main(config_file=args.c)  




















    
