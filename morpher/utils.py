import jsonpickle as jp
import inspect
import jsonpickle.ext.numpy as jsonpickle_numpy
import time

jsonpickle_numpy.register_handlers()

def pickle(obj, path=None):
	
	try:
		frozen = jp.encode(obj)
		if not path:
			path = retrieve_name(obj)
		with open(path,'w') as file:
			print(f"Pickling {path}...")
			file.write(frozen)
		return True
	except Exception as e:
		print(f"Could not pickle object. {e}")
		return False

def unpickle(path):
	try:
		with open(path, 'r') as file:
		    frozen = file.read()
		
		print(f"Unpickling {path}...")
		return jp.decode(frozen)

	except Exception as e:
		print(f"Could not unpickle object. {e}")
		return False

def retrieve_name(var):
    """
    Gets the name of var. Does it from the out most frame inner-wards.
    :param var: variable to get name from.
    :return: string
    """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]

class Timer:
	interval = 0    
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start     

