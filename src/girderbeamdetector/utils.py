import numpy as np

def to_list(arr):
    """[np.ndarrayをリストに変換する]

    Args:
        arr ([np.ndarray]): [numpy配列]

    Returns:
        [list]: [リスト]
    """    
    if not isinstance(arr, list):
        arr = arr.tolist()
    return arr

def to_lists(arr):
    """[np.ndarrayをリストに変換する。arrが2次元の場合、2次元目もlistに変換する。]

    Args:
        arr ([np.ndarray]): [numpy配列]

    Returns:
        [list]: [リスト]
    """    
    if len(arr) != 0:
        if isinstance(arr[0], list) or isinstance(arr[0], np.ndarray):
            arr = [to_list(elm) for elm in arr]
        else:
            arr = to_list(arr)
    else:
        arr = to_list(arr)
        
    return arr

def to_array(arr):
    """[リストをnp.ndarrayに変換する]

    Args:
        arr ([list]): [リスト型配列]

    Returns:
        [np.ndarray]: [numpy配列]
    """    
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    return arr