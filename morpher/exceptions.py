def kwarg_not_empty(arg, name):
    if not arg:
        raise AttributeError("Argument '%s' was not supplied." % name)
