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
def df_manipulation(df,col_selection=None,keep=None,subset=None):
    """
    1) Column selection: Keep columns in dataframe
    2) Data impute: Impute NA rows with empty string
    3) Data duplication cleaning: Drop all duplicates or drop all duplicates except for the first/last occurrence
    
    params:
    df [dataframe]: input dataframe     
    col_selection [None/list]: - None [Default]: Keep all columns in dataframe 
                               - List: List of columns to keep in dataframe                      
                                 
    keep[None/string/False]: Choose to drop all duplicates or drop all duplicates except for the first/last occurrence
                      # - None[DEFAULT] : Drop duplicates except for the first occurrence. 
                      # - "last" : Drop duplicates except for the last occurrence. 
                      # - False : Drop all duplicates.                 
    subset[list/None]: Subset of columns for identifying duplicates, use None if no column to select
   
    """
    logger.info("df_manipulation starts")
    logger.info("Shape of df before manipulation: %s",df.shape)

    #Column selection - user can select column(s) 
    if col_selection != None:
        df = df[col_selection]
    
    logger.info("Shape of df after selecting columns: %s",df.shape)

    #---Data impute - user can impute or drop rows with NA,freq of null values before & after manipulation returned---#
    logger.info("Number of null values in df:\n",df.isnull().sum())
  

    # impute NA values with empty string
    impute_value = ""
    df = df.fillna(impute_value)
    logger.info("Number of null values in df after NA imputation:\n %s",df.isnull().sum())
       
    #---------Data duplication cleaning--------#
    logger.info("Number of duplicates in the df: %s", df.duplicated().sum())

    #drop duplicates
    if keep==None:
        keep="first"
    df = df.drop_duplicates(subset=subset, keep=keep)

    logger.info("Shape of df after manipulation: %s",df.shape)
    logger.info("df_manipulation ends")

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
    characters[string]: input regex of characters to be removed  
    
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


def custom_taxo(df,remove_taxo,include_taxo):
    """
    User provides taxonomy to be removed or remained in the text. 
    a) user wants to remove taxonomies only -> input a list of taxonomies to be removed in remove_taxo 
    b) user wants to remove taxonomies but wants the same taxonomy to remain in certain phrases 
    (i.e remove taxo "test" but  "test" remains in "test cycle") -> input a list of taxonomies to be removed in remove_taxo and list of
    phrases for the taxonomy to remain in include_taxo
    
    params:
    df [dataframe]: input dataframe
    remove_taxo[list]: list of taxonomy to be removed from text
    include_taxo[list/None]: list of taxonomy to be maintained in text
    """
    logger.info("custom_taxo starts")
    
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
             for w in remove_taxo:
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

def stem_words(df,stemmer_type):
    """
    Stemming words. Default option is Porter Stemmer, alternative option is Lancaster Stemmer 
    params:
    df[dataframe]: input dataframe
    stemmer_type[None/string]: input stemming method 
                                - None for Porter Stemmer
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

def lemmatize_words(df,lemma_type):
    """
    Lemmatize words: Default option is WordNetLemmatizer, alternative option is Spacy 
    params:
    df[dataframe]: input dataframe
    lemma_type[None/string]: input lemmatization method
                            - None for WordNetLemmatizer
                            - "Spacy" for Spacy    
    """
    logger.info("lemmatize_words starts")    
    if lemma_type == None:
        logger.info("WordNetLemmatizer chosen for lemmatization")
        lemmatizer = WordNetLemmatizer()
        df = df.applymap(lambda text: " ".join([lemmatizer.lemmatize(word) for word in text.split()]))
        
    if lemma_type == "Spacy":
        logger.info("Spacy chosen for lemmatization")
        nlp = spacy.load("en_core_web_sm")
        df = df.applymap(lambda text: " ".join([word.lemma_ for word in nlp(text)]))
        #convert to lower case as spacy will convert pronouns to upper case
        df = df.applymap(lambda s:s.lower() if type(s) == str else s) 
        
    df = df.add_suffix('_lemma')
    logger.info("lemmatize_words ends")
    
    return df

def feature_extraction(column,ngram_range=None,ascending=None,fe_type=None):
    """
    Feature extraction methods - TF-IDF(default choice) or Bag of words
     
    params:
    column [series/DataFrame]: column selected for feature extraction 
                        - series: only one column is selected for feature extraction (e.g. df["title_clean"])
                        - DataFrame: more than one column is selected for feature extraction (e.g. df[["title_clean","desc_clean"]])
    ngram_range [tuple(min_n, max_n)]: The lower and upper boundary of the range of n-values for different n-grams to be extracted
                                       - [default] ngram_range of (1, 1) means only unigrams, 
                                       - ngram_range of (1, 2) means unigrams and bigrams, 
                                       - ngram_range of (2, 2) means only bigram
    ascending [True/False/None]: - [default] None (words arranged in alphabetical order)
                                 - True(words arranged in ascending order of sum), 
                                 - False(words arranged in descending order of sum)                               
    fe_type[string/None]: Feature extraction type: Choose "bagofwords" for bow or None for default tfidf method
    
    """
    logger.info("feature_extraction starts")        
    
    if type(column) == pd.DataFrame: #concat the columns into one string if there is more than one column 
        column = column.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        logger.info("The column(s) selected for feature extraction: %s", column) 
           
    if ngram_range == None: #set ngram range as unigram by default
        ngram_range=(1,1)
        
    if fe_type == "bagofwords":      
        logger.info("BagofWords selected as feature extraction method") 
        vec_type = CountVectorizer(ngram_range=ngram_range, analyzer='word')
        vectorized = vec_type.fit_transform(column)
        df = pd.DataFrame(vectorized.toarray(), columns=vec_type.get_feature_names())
        df.loc['sum'] = df.sum(axis=0).astype(int)

    if fe_type == None: #tfidf   
        logger.info("TF-IDF selected as feature extraction method")  
        vec_type = TfidfVectorizer(ngram_range=ngram_range, analyzer='word')
        vectorized = vec_type.fit_transform(column)
        df = pd.DataFrame(vectorized.toarray(), columns=vec_type.get_feature_names())
        df.loc['sum'] = df.sum(axis=0)
    
    if ascending != None:
            
        df = df.sort_values(by ='sum', axis = 1,ascending=ascending)
    
    logger.info("The column(s) selected for feature extraction: %s", column) 
    logger.info("ngram_range selected is %s",ngram_range)
    
    logger.info("feature_extraction ends")
    
    return df,vec_type,vectorized