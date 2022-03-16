import sys
import UsrIntel.R2
import pandas as pd    
import json
import logging  
import argparse
import os
import nltk
from dataloading import data_loading
from datapreprocessing import df_manipulation,df_filterrows,word_contractions,lowercase,remove_htmltag_url,remove_irrchar_punc,remove_num,remove_multwhitespace,remove_stopwords,remove_freqwords,remove_rarewords
from datapreprocessing import custom_remtaxo,custom_keeptaxo,stem_words,lemmatize_words,feature_extraction
from mlmodule import kmeans_clustering,lda,nmf,supervised_lng,deep_lng,cosinesimilarity,jaccardsimilarity

#read config file and call the other functions
def main(projname,config_file,path_config):          
   
    with open(config_file) as f:
        data = json.load(f) #load user's config file
             
    with open(path_config) as pc:
        path = json.load(pc) #load the path config file (for core team use only)
    
    try:
        data_path = path["data_path"]            
    except: #output printed in current working= directory if not specified
        data_path = os.path.abspath(os.getcwd())
    
    try:    
        log_path = path["log_path"]
    except:#logs printed in current working directory if not specified
        log_path = os.path.abspath(os.getcwd())
    
    log_path = log_path + str(projname) + '/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p', filename=log_path+str(projname)+"_logs.log", filemode='a')
    logger = logging.getLogger("MLTrack") 
    
    logger.info("########-----------------------------------------############")
    logger.info("PROJECT NAME: %s",str(projname))
    logger.info("########-----------------------------------------############")
    logger.info("Interface script is executing")
    
    logger.info("User and path config files loaded")
    logger.info("Data path: %s",data_path)
    logger.info("Log path: %s",log_path)
        
    if path["outpath"]["user"]["enable"]:#store output in user folder 
        logger.info("Store output in user folder")
        try:
            outpath = path["outpath"]["user"]["user_outpath"]            
        except:
            outpath = os.path.abspath(os.getcwd())
       
        # Create folders according to projname in outpath if they don't exist already
        outpath = outpath + str(projname) + '/'
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        
        logger.info("User Output Path: %s",outpath)
            
    if path["outpath"]["elk"]["enable"]: 
        logger.info("Store output in user and ELK folder")
        
        try:
            outpath_elk = path["outpath"]["elk"]["elk_outpath"]  #store unsupervised learning csv files in ELK folder
            outpath = path["outpath"]["user"]["user_outpath"]   #store output in user folder for checking 
        except:
            outpath_elk = os.path.abspath(os.getcwd()) 
            outpath = os.path.abspath(os.getcwd())
        
        # Create folders according to projname in outpath/outpath_elk if they don't exist already
        outpath = outpath + str(projname) + '/'
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        
        outpath_elk = outpath_elk + str(projname) + '/'
        if not os.path.exists(outpath_elk):
            os.makedirs(outpath_elk)
        
        logger.info("User Output Path: %s",outpath)
        logger.info("ELK Output Path: %s",outpath_elk)
    
                
   #set the path to search for the model packages from nltk and spacy
    # nltk.data.path.append(path["package_path"])
    nltk.data.path = [path["package_path"]]
                
        
    #---------DATA LOADING----------#
    logger.info("Data Loading starts")
    dl = data["DataLoading"]
            
    start_date,stop_date = dl['start_date'],dl['stop_date']  
    df = data_loading(path=data_path,start_date=start_date,stop_date=stop_date)
    logger.info("Data Loading ends")
    
           
    #---------DATA PREPROCESSING----------# 
    logger.info("Data Preprocessing starts")
    
    #df_filterrows
    fr = data["DataPreprocessing"]["df_filterrows"]
    if data["DataPreprocessing"]["df_filterrows"]["enable"]:
        col,keep_list,drop_list = fr["col"],fr["keep_list"],fr["drop_list"]
        df = df_filterrows(df,col=col,keep_list=keep_list,drop_list=drop_list)
        df_all = df.copy()
    
    #df_manipulation
    df_out = df.drop(["file"],axis=1) #all text -> ml output
    dm = data["DataPreprocessing"]["df_manipulation"]
    how,col_selection,keep,subset = dm['how'],dm['col_selection'],dm['keep'],dm['subset']  
    df = df_manipulation(df,how=how,col_selection=col_selection,keep=keep,subset=subset)
   
    df_all = df.copy() #id, raw text -> data preprocessing final file    
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
        
    # Custom remove taxonomy - User gives taxonomy to remove from data
    crt = data["DataPreprocessing"]["custom_remtaxo"]
    remtaxo,remove_taxo,include_taxo = crt["enable"], crt["remove_taxo"], crt["include_taxo"]
    if remtaxo:
        df = custom_remtaxo(df,remove_taxo,include_taxo)
        df_all = pd.concat([df_all,df],axis=1)  
    
    # Custom keep taxonomy - User gives taxonomy to keep in data
    ckt = data["DataPreprocessing"]["custom_keeptaxo"]
    keeptaxo,keep_taxo = ckt["enable"], ckt["keep_taxo"]
    if keeptaxo:
        df = custom_keeptaxo(df,keep_taxo)
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
          
    # Stemming
    st = data["DataPreprocessing"]["stem_words"]
    stem,stemmer_type = st["enable"],st["stemmer_type"]     
    if stem:
        df = stem_words(df,stemmer_type)
        df_all = pd.concat([df_all,df],axis=1)            
    
    #Lemmatization
    lem = data["DataPreprocessing"]["lemmatize_words"]
    lemma,lemma_type,package_path = lem["enable"], lem["lemma_type"],path["package_path"]      
    if lemma:
        df = lemmatize_words(df,package_path,lemma_type)
        df_all = pd.concat([df_all,df],axis=1)
    
    #column bind target to df
    if data["DataPreprocessing"]["target"]["enable"]:
        df = pd.concat([target,df],axis=1)
        df_all = pd.concat([target,df_all],axis=1) 
      
    df_all.to_excel(outpath+"preprocessed_text.xlsx",index=False) #save after data preprocessing   
    logger.info("Data Preprocessing ends")
    
    #---------------ML module---------------# 
    logger.info("Ml module starts")    
    
    
    ####---Unsupervised Learning---### 

    #k-means clustering 
    km= data["UnsupervisedLearning"]["kmeans_clustering"]
    kmeans = km["enable"]
    if kmeans:
        top_n_terms,ngram_range,fe_type,n_clusters,max_n_clusters,token_pattern= km["top_n_terms"],km["ngram_range"],km["fe_type"],km["n_clusters"],km["max_n_clusters"],km["token_pattern"]        
        df_out["cluster"]=kmeans_clustering(column=df,outpath=outpath,top_n_terms=top_n_terms,ngram_range=ngram_range,fe_type=fe_type,n_clusters=n_clusters,max_n_clusters=max_n_clusters,token_pattern=token_pattern)
        df_out.to_excel(outpath+"KMeansClustering_output.xlsx",index=False) #write to user folder  
        if path["outpath"]["elk"]["enable"]: #write to ELK folder
            df_out.to_excel(outpath_elk+"KMeansClustering_output.xlsx",index=False)
            logger.info("K-means clustering results saved in %s and %s as KMeansClustering_output.xlsx" %(outpath,outpath_elk)) 
        else:
            logger.info("K-means clustering results saved in %s as KMeansClustering_output.xlsx",outpath) 
        
    #LDA
    lda_m= data["UnsupervisedLearning"]["lda"]
    LatentDirichletAllocation = lda_m["enable"]
    if LatentDirichletAllocation:
        n_components,top_n_terms,ngram_range,token_pattern= lda_m["n_components"],lda_m["top_n_terms"],lda_m["ngram_range"],lda_m["token_pattern"]       
        df_out["cluster"]=lda(column=df,outpath=outpath,n_components=n_components,top_n_terms=top_n_terms,ngram_range=ngram_range,token_pattern=token_pattern)            
        df_out.to_excel(outpath+"LatentDirichletAllocation_output.xlsx",index=False)  #write to user folder 
        if path["outpath"]["elk"]["enable"]: #write to ELK folder
            df_out.to_excel(outpath_elk+"LatentDirichletAllocation_output.xlsx",index=False)
            logger.info("LDA results saved in %s and %s as LatentDirichletAllocation_output.xlsx" %(outpath,outpath_elk))             
        else:
            logger.info("LDA results saved in %s as LatentDirichletAllocation_output.xlsx",outpath) 

    #NMF Factorization
    nmf_m= data["UnsupervisedLearning"]["nmf"]
    NonNegativeMatrixFactorization = nmf_m["enable"]
    if NonNegativeMatrixFactorization:
        n_components,top_n_terms,fe_type,ngram_range,token_pattern= nmf_m["n_components"],nmf_m["top_n_terms"],nmf_m["fe_type"],nmf_m["ngram_range"],nmf_m["token_pattern"]
        df_out["cluster"]=nmf(column=df,outpath=outpath,n_components=n_components,top_n_terms=top_n_terms,fe_type=fe_type,ngram_range=ngram_range,token_pattern=token_pattern)              
        df_out.to_excel(outpath+"NonNegativeMatrixFactorization_output.xlsx",index=False) #write to user folder 
        if path["outpath"]["elk"]["enable"]: #write to ELK folder 
            df_out.to_excel(outpath_elk +"NonNegativeMatrixFactorization_output.xlsx",index=False)
            logger.info("NMF results saved in %s and %s as NonNegativeMatrixFactorization_output.xlsx" %(outpath,outpath_elk)) 
        else:
            logger.info("NMF results saved in %s as NonNegativeMatrixFactorization_output.xlsx",outpath) 
        
    
    #####---Similarity Metrics-----####
    #Cosine Similarity
    cs= data["SimilarityMetrics"]["cosinesimilarity"]
    cosinesim = cs["enable"]
    if cosinesim:
        threshold,total_rows,base_id,ngram_range,fe_type,ascending= cs["threshold"],cs["total_rows"],cs["base_id"],cs["ngram_range"],cs["fe_type"],cs["ascending"]        
        cosinesimilarity(column=df,identifier=df_all["id"],outpath=outpath,threshold=threshold,total_rows=total_rows,base_id=base_id,ngram_range=ngram_range,fe_type=fe_type,ascending=ascending)

    #Jaccard Similarity
    js= data["SimilarityMetrics"]["jaccardsimilarity"]
    jaccardsim = js["enable"]
    if jaccardsim:
        threshold,total_rows,base_id,ascending= js["threshold"],js["total_rows"],js["base_id"],js["ascending"]
        jaccardsimilarity(column=df,identifier=df_all["id"],outpath=outpath,threshold=threshold,total_rows=total_rows,base_id=base_id,ascending=ascending)

    #####-----Supervised Learning----####
             
    #Machine Learning
    sl= data["SupervisedLearning"]["supervised_lng"]
    SupervisedLearning = sl["enable"]
    if SupervisedLearning:
        target,test_size,ngram_range,fe_type,model_type,ascend= sl["target"],sl["test_size"],sl["ngram_range"],sl["fe_type"],sl["model_type"],sl["ascend"]
        supervised_lng(df=df,outpath=outpath,target=target,test_size=test_size,ngram_range=ngram_range,fe_type=fe_type,model_type=model_type,ascend=ascend)

    #Deep Learning
    dl= data["SupervisedLearning"]["deep_lng"]
    DeepLearning = dl["enable"]
    if DeepLearning:
        target,test_size,ngram_range,fe_type,hidden_layer_sizes,activation,solver,learning_rate,max_iter,ascend= dl["target"],dl["test_size"],dl["ngram_range"],dl["fe_type"],dl["hidden_layer_sizes"],dl["activation"],dl["solver"],dl["learning_rate"],dl["max_iter"],dl["ascend"]
        deep_lng(df=df,outpath=outpath,target=target,test_size=test_size,ngram_range=ngram_range,fe_type=fe_type,hidden_layer_sizes=hidden_layer_sizes,activation=activation,solver=solver,learning_rate=learning_rate,max_iter=max_iter,ascend=ascend)
    
    logger.info("Ml module ends")

    
if __name__ == '__main__':  
        
    parser = argparse.ArgumentParser()
    parser.add_argument('-s',type=str, required=True)
    parser.add_argument('-c',type=str, nargs = 2, required=True)
    
    args = parser.parse_args()
    
    main(projname = args.s,config_file = args.c[0],path_config = args.c[1])  




















    
