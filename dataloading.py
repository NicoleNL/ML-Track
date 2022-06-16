from datetime import datetime,timedelta
from jmespath import search
import pandas as pd
import glob, os, json
import re
import logging
logger = logging.getLogger("MLTrack")


def get_filenames(path:str) -> list:
    """Get all file names given the path(folder)

    Args:
        path (str): path(folder) to obtain files

    Returns:
        list: list of filenames
    """
    return [f"{path}/{f}" for f in os.listdir("data") if os.path.isfile(os.path.join("data", f))]
    
def get_json_filenames(path:str) -> list:
    """Get json only files given the path

    Args:
        path (str): path to obtain json files

    Returns:
        list: list of json files
    """
    pattern = r".*\.(json)$"
    return [f for f in get_filenames(path) if re.search(pattern, f)]

def load_json_lst(path:str) -> list:
    """Load list of json files into Dataframes

    Args:
        path (str): path to load json files

    Returns:
        list: _description_
    """
    return [pd.read_json(f) for f in get_json_filenames(path)]

def load_combine_json(path:str) -> pd.DataFrame:
    """Load and combine list of json DFs into a single DF

    Args:
        path (str): path to load and combine DFs

    Returns:
        pd.DataFrame: Combined DFs from path
    """
    return pd.concat(load_json_lst(path))
            
#data loading
def data_loading(path,start_date=None,stop_date=None):
    '''
    Load only files that follow agreed filename format, merge files as single dataframe.
    User can choose to 
    a) Load all json files following the agreed filename format
    b) Load only json files from specific dates by adding the start and stop dates (Note: Both start_date and
    stop_date must be used together)
    
    params:
    path [string]: path of the files, without filename
    
    start_date[None/string in YYYY-MM-DD format](optional,default is None): 
    User can choose to load files starting from start_date
    - None: no start_date is provided, all files are loaded
    - string in YYYY-MM-DD format: files starting from start_date will be loaded
    
    stop_date[None/string in YYYY-MM-DD format](optional,default is None): 
    User can choose to load files until stop_date
    - None: no stop_date is provided, all files are loaded
    - string in YYYY-MM-DD format: files until stop_date will be loaded
    '''
       
    filenames = os.listdir(path)
    file_list=[]
    date_list = []
    df = pd.DataFrame()
            
    if start_date == None and stop_date == None :
        for file in filenames:
            # search agreed file format pattern in the filename

            #pattern = r"^\(\d{4}-\d{2}-\d{1,2}\)\d+\_\D+\_\d+\.json$"
            pattern = r".*\.(json)$"

            match = re.search(pattern,file)
                
            #if match is found
            if match:
                pattern = os.path.join(path, file) #join path with file name
                file_list.append(pattern) #list of json files that follow the agreed filename
            
        logger.info("Files read: %s",file_list)                   
        for file in file_list:
            with open(file) as f:
                #flatten json into pd dataframe
                json_data = pd.json_normalize(json.loads(f.read()))
                json_data = pd.DataFrame(json_data)
                #label which file each row is from 
                json_data['file'] = file.rsplit("/", 1)[-1]

            df = df.append(json_data)              
                
    else:
        #convert start and stop string to datetime
        # logger.info("Load dataset from %s",start_date," until %s",stop_date)
        logger.info("Load dataset from %s until %s" %(start_date,stop_date))
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        stop = datetime.strptime(stop_date, "%Y-%m-%d").date()
    
        #iterate from start to stop dates by day and store dates in list
        while start <= stop:
            date_list.append(start)
            start = start + timedelta(days=1)  # increase day one by one

        #convert datetime objects to string
        string_list =[d.strftime("%Y-%m-%d") for d in date_list]
                
        for file in filenames: 
            
            # search agreed file format pattern in the filename
            for date in string_list: 
                pattern = r"\("+date+r"\)\d+\_\D+\_\d+\.json"
        
                match = re.search(pattern,file)
                
                #if match is found
                if match:
                    pattern = os.path.join(path, file) #join path with file name
                    file_list.append(pattern) #list of json files that follow the agreed filename

        logger.info("Files read: %s",file_list)     
        
        for file in file_list:
            with open(file) as f:
                #flatten json into pd dataframe
                json_data = pd.json_normalize(json.loads(f.read()))
                json_data = pd.DataFrame(json_data)
                #label which file each row is from 
                json_data['file'] = file.rsplit("/", 1)[-1]

            df = df.append(json_data)
            
    logger.info("Shape of dataframe loaded is %s",df.shape)    
    return df

