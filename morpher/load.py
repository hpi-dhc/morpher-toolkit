import traceback
import logging
from morpher.exceptions import kvarg_not_empty
import morpher.config as config
import pandas as pd
import numpy as np
import pyhdb

def load(source=config.FILE, **kvargs):

	data = pd.DataFrame()

	try:
		if (source == config.FILE):
			kvarg_not_empty(kvargs.get("file_name"),"file_name")
			data = load_from_file(kvargs.get("file_name"))

		elif (source == config.DATABASE):
			kvarg_not_empty(kvargs.get("sql"), "sql")
			data = load_from_server(kvargs.get("sql"))

	except Exception as e:
		logging.error(traceback.format_exc())

	return data


def load_from_server(sql):
    '''
    Opens a database connection to the HANA hosted Mimic3 db and pulls data.
    '''    
    try:

      connection = pyhdb.connect(host= config.db_address,
                                       port= config.db_port,
                                       user= config.db_user,
                                       password= config.db_password)
      with open(sql,'r') as f:
        query = f.read()
      cursor = connection.cursor()
      cursor = cursor.execute(query)
      data = pd.DataFrame(cursor.fetchall())      
    except Exception as e:
      print("*** Error connecting to DB. Check your environment variables.\n " + repr(e))
      raise e
    return data

def load_from_file(file):
    '''
    Opens a database connection to the HANA hosted Mimic3 db and pulls data.
    
    '''
    data = pd.read_csv(filepath_or_buffer= file)
    return data
