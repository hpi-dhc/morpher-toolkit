import jsonpickle as jp
import inspect

def pickle(obj, path=None):
	
	try:
		frozen = jp.encode(obj)
		if not path:
			path = retrieve_name(obj)
		with open(path,'w') as file:
			print("Pickling object...")
			file.write(frozen)
		return True
	except Exception as e:
		print(f"Could not pickle object. {e}")
		return False

def unpickle(path):
	try:
		with open(path, 'r') as file:
		    frozen = file.read()
		
		print("Unpickling object...")
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
