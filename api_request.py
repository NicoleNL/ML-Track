# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 10:06:52 2021

@author: nchong
"""

import requests

# USER CONFIGURABLE
projname= "IVE"   #specify project name to take path config file from project subfolder
configpath = "/tmp/dsabf/config/AIML/"+ str(projname) + "/config.json" # specify user config file path, to change to share drive path later

#DO NOT CHANGE
url = "http://172.30.212.7:5900/AIML?config=" + configpath + "&projname=" + projname
response = requests.get(url)
print(response.text)
print(response.status_code)

