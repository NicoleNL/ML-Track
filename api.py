# API for ML
import sys
import UsrIntel.R2
import subprocess
import logging
import sys
import ast
import os
import datetime
import configparser
from pathlib import Path
from flask import request, url_for, Flask, json


api = Flask(__name__)
   
# API End Points
@api.route("/AIML", methods=['GET'])
def api_ML():
    
    results = {'message': 'Invalid request'}
    if 'config' in request.args and 'projname' not in request.args: 
        results = {'message': 'Please specify the user config file path and project name.'}
    
    #Get user config file path and project name from user input
    CONFIG = str(request.args['config'])
    PROJNAME = str(request.args['projname'])
    
    #Logging for API
    log_path = "/tmp/dsabf/log/AIML/"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p', filename=log_path+"api_"+PROJNAME+"_logs.log", filemode='a')
    logger = logging.getLogger("AIML")  

    try:     
        logger.info("########-----------------------------------------############")
        logger.info("PROJECT NAME: %s",PROJNAME)
        logger.info("########-----------------------------------------############")
        logger.info("within try")
        
        # .py and config file
        INTERFACE = "interface.py"
        PATH_CONFIG = "/tmp/dsabf/config/AIML/" + PROJNAME + "/path_config.json"
        
        logger.info("Path config file is in %s",PATH_CONFIG)
        logger.info("User config file is in %s",CONFIG)
        
        command_line = [sys.executable, INTERFACE, "-s", PROJNAME, "-c", CONFIG, PATH_CONFIG]       
        
        logger.info('Subprocess command line: "' + str(command_line) + '"')
        
        logger.info('Executing python interface script') 
           
        command_line_process = subprocess.Popen(
            command_line,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT #standard error should go into the same handle as standard output
            
	    )
        process_output, _ =  command_line_process.communicate()
        #convert bytes to strings
        process_output = process_output.decode('utf-8')
                
        logger.info('Output: %s',process_output)        
        logger.info('Returncode %s', command_line_process.returncode)
                
        if command_line_process.returncode != 0:
            results = {'message': 'Process failed. Error: Issue with python script','returncode': command_line_process.returncode,'Output':process_output}
            logger.info('Subprocess failed. Error: Issue with python script. Returncode: %s',command_line_process.returncode)
            raise Exception	     
        else:
            logger.info('Finished executing python script')
            
        results = {'message': 'Process completed'}
        logger.info('Process completed')
                            
    except Exception as e:
        logger.info("Within except")
        logger.error('Exception occured: ' + str(e))        
        if results == {}:
           results = {'message': 'Process failed'}
           logger.info('Subprocess failed')   
           
    return json.dumps(results)

# Web service API
if __name__ == "__main__":
    api.run(host='0.0.0.0', port=5900, debug=False)

