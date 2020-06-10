import inspect


def get_class_by_name(classes_dict, class_name, default):
    try:
        class_by_name = classes_dict[class_name]
    except:
        try:
            class_by_name = classes_dict[class_name.split('_')[0]]
        except:
            print('Warning: using default, ', default)
            class_by_name = default
    return class_by_name


def get_default_args_of_function(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

