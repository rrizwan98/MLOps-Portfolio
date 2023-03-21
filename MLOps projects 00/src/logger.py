import logging
import os
from datetime import datetime

# creating a log folder to track all the execuation 
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log" 
logs_path = os.path.join(os.getcwd(),"logs",LOG_FILE) # folder created (logs) in a main directory
os.makedirs(logs_path,exist_ok=True)

LOG_FILEP_PATH = os.path.join(logs_path,LOG_FILE) # every execuation is record in new folder inside log folder

logging.basicConfig(
    filename = LOG_FILEP_PATH,
    format="[%(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s", # log info save this formate
    level=logging.INFO
)

