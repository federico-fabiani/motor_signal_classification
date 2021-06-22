import os


def clean_logs(log_path, keep=5):
    logs = os.listdir(log_path)
    while len(logs) > keep:
        os.remove(log_path + '/' + logs[0])
        logs.pop(0)


def f_unpack_dict(dct):
    """
    Unpacks all sub-dictionaries in given dictionary recursively. There should be no duplicated keys
    across all nested subdictionaries, or some instances will be lost without warning

    Parameters:
    ----------------
    dct : dictionary to unpack

    Returns:
    ----------------
    : unpacked dictionary
    """

    res = {}
    for (k, v) in dct.items():
        if isinstance(v, dict):
            res = {**res, **f_unpack_dict(v)}
        else:
            res[k] = v

    return res


# def f_wrap_space_eval(hp_space, trial):
#     """
#     Utility function for more consise optimization history extraction
#
#     Parameters:
#     ----------------
#     hp_space : hyperspace from which points are sampled
#     trial : hyperopt.Trials object
#
#     Returns:
#     ----------------
#     : dict(
#         k: v
#     ), where k - label of hyperparameter, v - value of hyperparameter in trial
#     """
#
#     return space_eval(hp_space, {k: v[0] for (k, v) in trial['misc']['vals'].items() if len(v) > 0})
