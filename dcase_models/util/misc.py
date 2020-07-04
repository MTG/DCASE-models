# encoding: utf-8
"""Miscellaneous functions"""

import inspect


def get_class_by_name(classes_dict, class_name, default):
    """ Get a class given its name.

    Parameters
    ----------
    classes_dict : dict
        Dict with the form {class_name: class}
    class_name : str
        Class name.
    default: class
        Class to be used if class_name is not in classes_dict.

    Returns
    -------
    Class
        Class with name class_name

    """
    try:
        class_by_name = classes_dict[class_name]
    except Exception:
        try:
            class_by_name = classes_dict[class_name.split('_')[0]]
        except Exception:
            print('Warning: using default, ', default)
            class_by_name = default
    return class_by_name


def get_default_args_of_function(func):
    """ Get default arguments of a function

    Parameters
    ----------
    func : function
        Function to be inspected.

    Returns
    -------
    Dict
        Dictionary with the function default arguments.

    """
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }
