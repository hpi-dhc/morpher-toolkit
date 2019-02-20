import logging

def kvarg_not_empty(arg, name):
	if not arg:
		raise AttributeError("Argument '{%}' was not supplied." % name)