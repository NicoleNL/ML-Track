from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem import PorterStemmer,LancasterStemmer,WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import contractions
import pandas as pd
import re 
import nltk
import spacy
import logging
logger = logging.getLogger("MLTrack")


#data preprocessing
def df_manipulation(df,how=None,col_selection=None,keep=None,subset=None):
    """
    1) Column selection: Keep columns in dataframe
    2) Data impute: Impute NA rows with empty string
    3) Data duplication cleaning: Drop all duplicates or drop all duplicates except for the first/last occurrence
    
    params:
    df [dataframe]: input dataframe  
    how[None/string]: Drop rows when we have at least one NA or all NA. Choose
                      # - None : NA imputed with empty string
                      # - "all": Drop row with all NA
                      # - "any": Drop row with at least one NA
    col_selection [None/list]: - None [Default]: Keep all columns in dataframe 
                               - List: List of columns to keep in dataframe                      
                                 
    keep[None/string/False]: Choose to drop all duplicates or drop all duplicates except for the first/last occurrence
                      # - None[DEFAULT] : Drop duplicates except for the first occurrence. 
                      # - "last" : Drop duplicates except for the last occurrence. 
                      # - False : Drop all duplicates.                 
    subset[list/None]: list: Subset of columns for dropping NA and identifying duplicates, 
                       None[DEFAULT]: if no column to select
   
    """
    
    logger.info("Shape of df before manipulation: %s",df.shape)

    #Column selection - user can select column(s) 
    if col_selection != None:
        df = df[col_selection]
    
    logger.info("Shape of df after selecting columns: %s",df.shape)

    #---Data impute - user can impute or drop rows with NA,freq of null values before & after manipulation returned---#
    logger.info("Number of null values in df:\n %s",df.isnull().sum())         
    
    if how == None: # impute NA values with empty string
        logger.info("NA is imputed with empty string")
        impute_value = ""
        df = df.fillna(impute_value)
        logger.info("Number of null values in df after NA imputation:\n %s",df.isnull().sum())        
    else: # drop rows with NA values
        logger.info("NA is dropped")
        df= df.dropna(axis=0, how=how,subset=subset)
        logger.info("Number of null values in df after dropping NA rows:\n %s",df.isnull().sum())
        logger.info("Shape of df after dropping NA rows:%s",df.shape)

    #---------Data duplication cleaning--------#
    logger.info("Number of duplicates in the df:%s", df.duplicated().sum())

    #drop duplicates
    if keep==None:
        keep="first"
    df = df.drop_duplicates(subset=subset, keep=keep)

    logger.info("Shape of df after manipulation:%s",df.shape)

    return df
       
    
def df_filterrows(df,col,keep_list,drop_list):
    """
    User can choose to keep or drop row(s) in a column by providing the list of elements in the row(s)
    params:
    df [dataframe]: input dataframe 
    col[string]: input column name in which the row(s) within are to be kept/dropped
    keep_list[list/None]: list - input list of rows to be kept, None - when using drop_list
    drop_list[list/None]: list - input list of rows to be dropped, None - when using keep_list
    Example: To keep/drop the rows ['hw.sa', 'fw.qcode'] in the column “component_affected"
    """
    col = str(col)
    
    if keep_list != None:
        logger.info("Distribution of column %s : \n", col)
        logger.info("%s \n",df[col].value_counts())        
        df = df[df[col].isin(keep_list)]
        logger.info("Distribution of column %s after retaining %s only:\n" %(col,keep_list))        
        logger.info("%s \n",df[col].value_counts()) 
        
    if drop_list != None:
        logger.info("Distribution of column %s : \n", col)    
        logger.info("%s \n",df[col].value_counts()) 
        df = df[~df[col].isin(drop_list)]
        logger.info("Distribution of column %s after dropping %s only:\n" %(col,drop_list)) 
        logger.info("%s \n",df[col].value_counts())     
        
    return df
    
def word_contractions(df):
    """
    Expand word contractions (i.e. "isn't" to "is not")
    params:
    df [dataframe]: input dataframe 
    """
    logger.info("word_contractions starts")
    df = df.applymap(lambda text: " ".join([contractions.fix(word) for word in text.split()]))
    df.rename(columns={'oldName1': 'newName1', 'oldName2': 'newName2'}, inplace=True)
    
    df = df.add_suffix('_cont')
    
    logger.info("word_contractions ends")
    
    return df

def lowercase(df):
    """
    Convert all characters to lower case
    param:
    df[dataframe]: input dataframe
    
    """
    logger.info("lowercase starts")
    df = df.applymap(lambda s:s.lower() if type(s) == str else s)
    df = df.add_suffix('_lower')
    logger.info("lowercase ends")
        
    return df 

def remove_htmltag_url(df):
    """
    Remove html tag and url
    params:
    df [dataframe]: input dataframe 
    
    """
    logger.info("remove_htmltag_url starts")
    df = df.applymap(lambda text:BeautifulSoup(text, 'html.parser').get_text(separator= " ",strip=True))
    logger.info("html tag removed")
    #remove url
    df = df.replace('https?[://%]*\S+',' ', regex=True)    
    logger.info("url removed")
    df = df.add_suffix('_tagrem')
    logger.info("remove_htmltag_url ends")
    
    return df

def remove_irrchar_punc(df,char=None):
    """
    Remove irrelevant characters and punctuation. Optional: User can specify special characters to be removed in regex
    format.    
    params:    
    df [dataframe]: input dataframe 
    char[string/None]: input regex of characters to be removed  
    
    """
    logger.info("remove_irrchar_punc starts")
    if char != None:
        #Remove special characters given by user
        df = df.replace(char,' ',regex = True)
        logger.info("%s is the character(s) removed",char)
        
    # Remove utf-8 literals (i.e. \\xe2\\x80\\x8)
    df = df.replace(r'\\+x[\d\D][\d\D]',' ',regex = True)
        
    #Remove special characters and punctuation
    df = df.replace('[^\w\s]',' ',regex = True)
    df = df.replace(r'_',' ',regex = True)
    
    df = df.add_suffix('_puncrem')
    logger.info("remove_irrchar_punc ends")
    
    return df

def remove_num(df):
    """
    Remove numeric data
    params:
    df [dataframe]: input dataframe 
    
    """
    logger.info("remove_num starts")
    df=df.replace('\d+',' ', regex=True) 
    df = df.add_suffix('_numrem')
    logger.info("remove_num ends")
    
    return df 

def remove_multwhitespace(df):
    """
    Remove multiple white spaces
    params:
    df [dataframe]: input dataframe 
    
    """
    logger.info("remove_multwhitespace starts")
    df = df.replace(' +',' ', regex=True)
    df = df.add_suffix('_wsrem')
    logger.info("remove_multwhitespace ends")
    
    return df


def remove_stopwords(df,extra_sw=None,remove_sw=None):
    """
    Removes English stopwords. Optional: user can add own stopwords or remove words from English stopwords  
    params:
    df [dataframe]: input dataframe 
    extra_sw [list] (optional): list of words/phrase to be added to the stop words 
    remove_sw [list] (optional): list of words to be removed from the stop words 
    """
    logger.info("remove_stopwords starts")
    #nltk.download('stopwords')
    all_stopwords = stopwords.words('english')
    
    #default list of stopwords
    if extra_sw == None and remove_sw==None:
        all_stopwords = all_stopwords
        logger.info("Default list of stopwords used")
        
    # add more stopwords
    elif remove_sw == None:
        all_stopwords.extend(extra_sw) #add to existing stop words list
        logger.info("%s added to existing stop words list",extra_sw)
        
    # remove stopwords from existing sw list
    elif extra_sw == None:
        all_stopwords = [e for e in all_stopwords if e not in remove_sw] #remove from existing stop words list
        logger.info("%s removed from existing stop words list",remove_sw)
        
    # remove and add stopwords to existing sw list
    else:
        all_stopwords.extend(extra_sw) #add to existing stop words list
        all_stopwords = [e for e in all_stopwords if e not in remove_sw] #remove from existing stop words list
        logger.info("%s added to existing stop words list",extra_sw)
        logger.info("%s removed from existing stop words list",remove_sw)
  
    for w in all_stopwords:
        pattern = r'\b'+w+r'\b'
        df = df.replace(pattern,' ', regex=True)
    
    logger.info("Stopwords that are removed: %s", all_stopwords)
    df = df.add_suffix('_stoprem')
    logger.info("remove_stopwords ends")
               
    return df 

def remove_freqwords(df,n):
    """
    Remove n frequent words
    params:
    df [dataframe]: input dataframe 
    n [integer]: input number of frequent words to be removed
    # """
    logger.info("remove_freqwords starts")
    cnt = Counter()
    for i in df:
    
        for text in df[i].values:
            for word in text.split():
                cnt[word] += 1
           
    #custom function to remove the frequent words             
    FREQWORDS = set([w for (w, wc) in cnt.most_common(n)])
    
    logger.info("Frequent words that are removed: %s", set([(w, wc) for (w, wc) in cnt.most_common(n)]))
    df = df.applymap(lambda text: " ".join([word for word in str(text).split() if word not in FREQWORDS]))
    df = df.add_suffix('_freqrem')
    logger.info("remove_freqwords ends")
    
    return df

def remove_rarewords(df,n):
    """
    Remove n rare words
    params:
    df [dataframe]: input dataframe 
    n [integer]: input number of rare words to be removed
    """
    logger.info("remove_rarewords starts")
    cnt = Counter()
    for i in df:
    
        for text in df[i].values:
            for word in text.split():
                cnt[word] += 1
           
    #custom function to remove the frequent words             
    RAREWORDS = set([w for (w, wc) in cnt.most_common()[:-n-1:-1]])
    
    logger.info("Rare words that are removed: %s", set([(w,wc) for (w, wc) in cnt.most_common()[:-n-1:-1]]))
    df = df.applymap(lambda text: " ".join([word for word in str(text).split() if word not in RAREWORDS]))
    df = df.add_suffix('_rarerem')
    logger.info("remove_rarewords ends")
    
    return df

def custom_remtaxo(df,remove_taxo,include_taxo):
    """
    User provides taxonomy to be removed from the text. 
    a) user wants to remove taxonomies only -> input a list of taxonomies or regex to be removed in remove_taxo 
    b) user wants to remove taxonomies but wants the same taxonomy to remain in certain phrases 
    (i.e remove "test" from text but want "test" to remain in "test cycle") -> input a list of taxonomies or regex to be removed in remove_taxo 
    and list of phrases for the taxonomy to remain in include_taxo
    
    params:
    df [dataframe]: input dataframe
    remove_taxo[list/regex]: list of taxonomies or regex(i.e. r'test \w+') to be removed from text 
    include_taxo[list/None](optional): list of phrases for the taxonomy to remain in 
    """
    
    import re
    import pandas as pd 
    logger.info("custom_taxo starts")
    def convert(text,remove_taxo):  
        """
        Uses regex given in remove_taxo to find and return all matches 
        """
        match = re.findall(remove_taxo,text)
        if match:                 
            new_row = {'Match':match}
            return(new_row)
        
    #if remove_taxo is regex call convert function to get all matches as a list
    if type(remove_taxo) == str: 
        # logger.info("User input the regex:",remove_taxo)
        logger.info("User input the regex:"+ remove_taxo)
        cv_list = []
        for i in range(len(df.columns)):
            for text in df.iloc[:,i]:
                cv = convert(text,remove_taxo)
                if cv:
                    cv_list.append(cv)
        #             print(cv_list)

        cv_df = pd.DataFrame(cv_list)
        remove_taxo = list(cv_df["Match"].apply(pd.Series).stack().unique())
        logger.info("Remove_taxo_list: %s",remove_taxo)
        
    def taxo(text,remove_taxo,include_taxo): 
        if remove_taxo != None and include_taxo != None: #user wants to remove taxonomies but wants the same taxonomy to remain in certain phrases (i.e remove "test" but remain "test" in "test cyccle")

            for w in remove_taxo:
            #row without any item from include_taxo -> replace all remove_taxo items with empty string
                if all(phrase not in text for phrase in include_taxo): 
                    pattern = r'\b'+w+r'\b'
                    text = re.sub(pattern,' ', text) 
                #row with any item from include_taxo -> only replace remove_taxo item that is not in include_taxo
                else: 
                    if all(w not in phrase for phrase in include_taxo):
                        pattern = r'\b'+w+r'\b'
                        text = re.sub(pattern,' ', text) 
                        
        if remove_taxo != None and include_taxo == None: #user wants to remove taxonomies only:
            for w in remove_taxo: #remove_taxo in list of words
                pattern = r'\b'+w+r'\b'
                text = re.sub(pattern,' ', text)
                 
        return text 
    
    
    df = df.applymap(lambda text: taxo(text,remove_taxo,include_taxo))     
    df = df.add_suffix('_taxo')
     
    if remove_taxo != None and include_taxo != None:
        logger.info("User wants to remove taxonomies but wants the same taxonomy to remain in certain phrases")
        logger.info("Taxonomies removed: %s",remove_taxo)
        logger.info("Taxonomies remain in phrases: %s",include_taxo) 
    if remove_taxo != None and include_taxo == None: 
        logger.info("user wants to remove taxonomies only")
        logger.info("Taxonomies removed: %s",remove_taxo)
        
    logger.info("custom_taxo ends")
           
    return df    

def custom_keeptaxo(df,keep_taxo):
    """
    User provides taxonomy to be kept in the text, the rest of the taxonomy will be omitted. User can choose to do one 
    or multi stage filtering on the taxonomy
    
    df[DataFrame]: input dataframe
    keep_taxo[regex/list of regex]: use regex/list of regex to filter the taxonomy to keep
        regex : only one regex in keep_taxo to filter the taxonomy; i.e "[\w+\.]+\w+@intel.com"
        list of regex: use list of regex to filter the taxonomy for multi stage filter 
        i.e. ["[\w+\.]+\w+@intel.com","@intel.com","@intel","intel"]
        
    """        
    import re
    import pandas as pd 
    logger.info("custom_keeptaxo starts")
          
    logger.info("User input the regex:" + str(keep_taxo))
    if type(keep_taxo) == str:
        logger.info("Single stage filtering of taxonomy selected")
    else:
        logger.info("Multi stage filtering of taxonomy selected")
        
    
    def taxo_tokeep(text,keep_taxo):
        """
        Convert the taxo in keep_taxo to string and return the string  
        """
        #keep_taxo given as regex, get the list of words to keep in text according to regex
        if type(keep_taxo) == str:             
            keep_taxo = re.findall(keep_taxo,text)
            text = " ".join(str(word) for word in keep_taxo) # convert list to string    
        else:
            #loop through list of regex for multi stage filtering on the text 
            match_list = []    
            keep_taxo_list = re.findall(keep_taxo[0],text) #for first regex, get all the matches in a list
#             logger.info("Matches for first regex:", keep_taxo_list)
            #for second regex onwards, check each term in keep_taxo_list whether it matches the second regex    
            for tax in keep_taxo[1:]: 
                for i in keep_taxo_list:            
                    match = re.match(tax,i) #use match instead of findall which does not work with negation regex
                    if match:                         
                        match_list.append(match[0])
                
                text = " ".join(str(word) for word in match_list) # convert list to string  
                                
        return text    
        
    df = df.applymap(lambda text: taxo_tokeep(text,keep_taxo)) 
    df = df.add_suffix('_keeptaxo')
    
    logger.info("Get the taxonomies maintained in the texts")
    df_list =[]
    for col in df:        
        df_list.extend(list(df[col]))
    
    while("" in df_list) :
        df_list.remove("")
        
    logger.info("The taxonomies maintained in the texts: %s",list(set(df_list)))    
    logger.info("custom_keeptaxo ends")
           
    return df    

def stem_words(df,stemmer_type):
    """
    Stemming words. Default option is Porter Stemmer, alternative option is Lancaster Stemmer 
    params:
    df[dataframe]: input dataframe
    stemmer_type[None/string]: input stemming method 
                                - None for Porter Stemmer [DEFAULT]
                                - "Lancaster" for Lancaster Stemmer 
    """
    logger.info("stem_words starts")    
    if stemmer_type == None:
        logger.info("PorterStemmer chosen for stemming")
        stemmer = PorterStemmer()
    if stemmer_type == "Lancaster":
        logger.info("LancasterStemmer chosen for stemming")
        stemmer=LancasterStemmer()
    df = df.applymap(lambda text: " ".join([stemmer.stem(word) for word in text.split()]))
    df = df.add_suffix('_stem')
    logger.info("stem_words ends")
    
    return df

def lemmatize_words(df,package_path,lemma_type):
    """
    Lemmatize words: Default option is WordNetLemmatizer, alternative option is Spacy 
    params:
    df[dataframe]: input dataframe
    lemma_type[None/string]: input lemmatization method
                            - None for WordNetLemmatizer [DEFAULT]
                            - "Spacy" for Spacy    
    """
    logger.info("lemmatize_words starts")    
    if lemma_type == None:
        logger.info("WordNetLemmatizer chosen for lemmatization")
       # nltk.download('wordnet')
        lemmatizer = WordNetLemmatizer()
        df = df.applymap(lambda text: " ".join([lemmatizer.lemmatize(word) for word in text.split()]))
        
    if lemma_type == "Spacy":
        logger.info("Spacy chosen for lemmatization")               
        nlp = spacy.load(package_path+'en_core_web_sm-3.2.0/en_core_web_sm/en_core_web_sm-3.2.0')
        df = df.applymap(lambda text: " ".join([word.lemma_ for word in nlp(text)]))
        #convert to lower case as spacy will convert pronouns to upper case
        df = df.applymap(lambda s:s.lower() if type(s) == str else s) 
        
    df = df.add_suffix('_lemma')
    logger.info("lemmatize_words ends")
    
    return df

def feature_extraction(column,ngram_range=None,ascending=None,fe_type=None,token_pattern = None):
    """
    Feature extraction methods - TF-IDF[DEFAULT] or Bag of words
     
    params:
    column [series/DataFrame]: column selected for feature extraction 
                        - series: only one column is selected for feature extraction (e.g. df["title_clean"])
                        - DataFrame: more than one column is selected for feature extraction (e.g. df[["title_clean","desc_clean"]])
    ngram_range [tuple(min_n, max_n)]: The lower and upper boundary of the range of n-values for different n-grams to be extracted
                                       - [DEFAULT] ngram_range of (1, 1) means only unigrams, 
                                       - ngram_range of (1, 2) means unigrams and bigrams, 
                                       - ngram_range of (2, 2) means only bigram
    ascending [True/False/None]: - [DEFAULT] None (words arranged in alphabetical order)
                                 - True(words arranged in ascending order of sum), 
                                 - False(words arranged in descending order of sum)                               
    fe_type[string/None]: Feature extraction type: Choose "bagofwords" for bow or None for default tfidf method
    token_pattern[regex/None]: None: default regexp select tokens of 2 or more alphanumeric characters 
                               (punctuation is completely ignored and always treated as a token separator).
                               regex: Regular expression denoting what constitutes a “token"
    """
    logger.info("feature_extraction starts")        
    
    if type(column) == pd.DataFrame: #concat the columns into one string if there is more than one column 
        column = column.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)         
    
        
    if ngram_range == None: #set ngram range as unigram by default
        ngram_range=(1,1)
        
    if fe_type == "bagofwords":      
        logger.info("BagofWords selected as feature extraction method") 
        if token_pattern == None:
            logger.info("Default token pattern used")
            vec_type = CountVectorizer(ngram_range=ngram_range, analyzer='word')
        else:
            logger.info("Token pattern used to split the text in feature extraction: %s", token_pattern)
            vec_type = CountVectorizer(ngram_range=ngram_range, analyzer='word',token_pattern = token_pattern)
            
        vectorized = vec_type.fit_transform(column)
        df = pd.DataFrame(vectorized.toarray(), columns=vec_type.get_feature_names())
        df.loc['sum'] = df.sum(axis=0).astype(int)

    if fe_type == None: #tfidf   
        logger.info("TF-IDF selected as feature extraction method")  
        if token_pattern == None:
            logger.info("Default token pattern used")
            vec_type = TfidfVectorizer(ngram_range=ngram_range, analyzer='word')
        else:
            logger.info("Token pattern used to split the text in feature extraction: %s", token_pattern)
            vec_type = CountVectorizer(ngram_range=ngram_range, analyzer='word',token_pattern = token_pattern)
            
        vectorized = vec_type.fit_transform(column)
        df = pd.DataFrame(vectorized.toarray(), columns=vec_type.get_feature_names())
        df.loc['sum'] = df.sum(axis=0)
    
    if ascending != None:
            
        df = df.sort_values(by ='sum', axis = 1,ascending=ascending)
    
    
    logger.info("ngram_range selected is %s",ngram_range)
    
    logger.info("feature_extraction ends")
    
    return df,vec_type,vectorized


### Additional code####
### Custom taxonomy with NER###
# import pandas as pd
# from tqdm import tqdm
# import spacy
#from spacy.tokens import DocBin
# import numpy as np

# def convert_spacy(DATA):
#     """
#     Convert  data into .spacy format
#     DATA[]: Train/validation data to be converted to .spacy format
#     """
#     nlp = spacy.blank("en") # load a new spacy model
#     db = DocBin() # create a DocBin object

#     for text, annot in tqdm(DATA): # data in previous format
#         doc = nlp.make_doc(text) # create doc object from text
#         ents = []
#         for start, end, label in annot["entities"]: # add character indexes
#             span = doc.char_span(start, end, label=label, alignment_mode="contract")
#             if span is None:
#                 print("Skipping entity")
#             else:
#                 ents.append(span)
#         doc.ents = ents # label the text with the ents
#         db.add(doc)
        
#     return db

    
# def custom_ner(TRAIN_DATA,VAL_DATA,path):
#     """
#     Build and save custom NER model in given path. 
    
#     """
#     #convert train and validation data into .spacy format
#     db_train = convert_spacy(TRAIN_DATA) 
#     db_val = convert_spacy(VAL_DATA) 
    
#     #save train and validation data in .spacy format in path
#     db_train.to_disk(path +'train.spacy')
#     db_val.to_disk(path +'val.spacy')
    
#     print("Train and validation converted to .spacy format and saved")
    
#     #autofill base_config file saved by user from spacy website
#     !python -m spacy init fill-config base_config.cfg config.cfg
    
#     #Model building and saving in path
#     !python -m spacy train config.cfg --output ./output --paths.train ./train.spacy --paths.dev ./val.spacy
    
#     print("Custom NER model built and saved!")
    
# def check_ents(path,column):
#     """
#     Check entities after loading best model
    
#     """
#     #Load best model
#     nlp = spacy.load(path + "/output/model-best/")     
#     print("Best model loaded!")
    
#     entities = []
#     for text in column.tolist():
#         doc = nlp(text)
#         for ent in doc.ents:
#             entities.append(ent.text+' - '+ent.label_)
#     print(np.unique(np.array(entities)))        

# def ner_wrapper(TRAIN_DATA,VAL_DATA,path,column,train_model):  
#     """
#     User can choose to train the spacy model or load spacy model
#     params:
#     TRAIN_DATA[NER format]: train data for model building
#     VAL_DATA[NER format]: validation data for model building
#     path[string]: input path to store model. Path has to be the same as base_config.cfg file downloaded from spacy
#                   website and jupyter notebook.
#     column[series]: column for entities to be checked
#     train_model[True/False]: True if want to train model. False to load model (no training)
#     """
#     if train_model == True:
#         custom_ner(TRAIN_DATA,VAL_DATA,path)
#         check_ents(path,column)
        
#     if train_model == False:
#         check_ents(path,column)

# ### Custom Tokenization ###
# def cust_tokenization_split(column,delim =None):
#     """
#     Custom tokenization using split() 
#     params:
#     column[series]: input column           
#     delim[None/string],default delimiter (delim=None) is whitespace: specify delimiter to separate strings
#                         - None: delimiter is white space
#                         - string: delimiter is the string specified       
#     """
    
#     if delim==None:
#         print("Text is split by whitespace") #default delimiter is space if not specified 

#     else:
#         print("Text is split by:", delim) #can accept one or more delimiter

#     return column.apply(lambda text: text.split() if delim==None else text.split(delim))

# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.tokenize import sent_tokenize
# from nltk.tokenize import WhitespaceTokenizer
# from nltk.tokenize import WordPunctTokenizer

# def cust_tokenization_nltk(column,token_type):
#     """
#     Custom tokenization using NLTK 
#     params:
#     column[series]: input column 
#     token_type["string"]: type of nltk tokenization
#     a) token_type = "WordToken" tokenizes a string into a list of words
#     b) token_type = "SentToken" tokenizes a string containing sentences into a list of sentences
#     c) token_type = "WhiteSpaceToken" tokenizes a string on whitespace (space, tab, newline)
#     d) token_type = "WordPunctTokenizer" tokenizes a string on punctuations
#     """
#     if token_type == "WordToken":
#         tokenizer = word_tokenize
#     if token_type == "SentToken":
#         tokenizer = sent_tokenize
#     if token_type == "WhiteSpaceToken":
#         tokenizer = WhitespaceTokenizer().tokenize
#     if token_type == "WordPunctTokenizer":
#         tokenizer = WordPunctTokenizer().tokenize

#     return column.apply(lambda text: tokenizer(text))
