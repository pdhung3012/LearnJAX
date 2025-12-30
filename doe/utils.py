import json
import random
import os
from typing import Any, Dict, List, Optional, Tuple, Union

JSONObj = Dict[str, Any]
Path = Tuple[Union[str, int], ...]

def ensure_dir_os(path: str) -> str:
    os.makedirs(path, exist_ok=True)  # no error if it already exists
    return path

def find_largest_list_json(
    obj_or_str: Union[str, JSONObj],
    *,
    return_path: bool = False
) -> Optional[Union[
    List[Any],
    Tuple[str, List[Any]],
    Tuple[str, List[Any], Path]
]]:
    """
    Search a JSON object (string or dict) for the list with the most elements.
    Returns the list; and also the nearest dict key owning that list.

    Returns:
        - If return_path=False:
            * list_only                        -> List[Any]
            * with key                         -> (key: str|None, list: List[Any])
        - If return_path=True:
            * with key and path                -> (key: str|None, list: List[Any], path: Path)
        - None if no list exists anywhere.

    Notes:
        - 'key' is the closest dict key whose value is (or contains at that node) the found list.
        - If the list is nested only under arrays with no intervening dict key, key will be None.
    """
    # Parse if needed
    if isinstance(obj_or_str, str):
        try:
            root = json.loads(obj_or_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}") from e
        if not isinstance(root, dict):
            raise ValueError("Top-level JSON must be an object (dict).")
    elif isinstance(obj_or_str, dict):
        root = obj_or_str
    else:
        raise TypeError("Input must be a JSON string or a dict.")

    best: Optional[Tuple[List[Any], Path]] = None

    def nearest_key(path: Path) -> Optional[str]:
        # Return the last string element in the path (nearest dict key).
        for p in reversed(path):
            if isinstance(p, str):
                return p
        return None

    def consider(candidate: List[Any], path: Path):
        nonlocal best
        if best is None or len(candidate) > len(best[0]):
            best = (candidate, path)

    def walk(node: Any, path: Path):
        if isinstance(node, list):
            consider(node, path)
            for i, item in enumerate(node):
                walk(item, path + (i,))
            return
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, path + (k,))
            return
        # primitives: nothing to do

    walk(root, ())

    if best is None:
        return None

    the_list, path = best
    key = nearest_key(path)

    if return_path:
        return key, the_list, path
    # If caller wants both the key and the list (but not the path), return a pair.
    return [key, the_list]



def find_first_list_json(obj_or_str: Union[str, JSONObj], *, return_path: bool = False
                        ) -> Optional[Union[List[Any], Tuple[List[Any], Path]]]:
    """
    Accepts a JSON string or a dict (parsed JSON object). Performs a depth-first, pre-order
    scan of dict values and returns the first value whose type is `list`.

    Args:
        obj_or_str: JSON string (top-level must be an object) or a Python dict.
        return_path: If True, also return the access path (tuple of keys) to the found list.

    Returns:
        - If return_path=False: the first list found, or None if none exists.
        - If return_path=True: (the_list, path) or None if none exists.

    Raises:
        ValueError: if given a string that is not valid JSON or whose top-level is not an object.
        TypeError:  if given a non-dict, non-string input.
    """
    # Load if a string
    if isinstance(obj_or_str, str):
        try:
            root = json.loads(obj_or_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}") from e
        if not isinstance(root, dict):
            raise ValueError("Top-level JSON must be an object (dict).")
    elif isinstance(obj_or_str, dict):
        root = obj_or_str
    else:
        raise TypeError("Input must be a JSON string or a dict (parsed JSON object).")

    def walk(node: Any, path: Path) -> Optional[Tuple[List[Any], Path]]:
        if isinstance(node, list):
            return node, path
        if isinstance(node, dict):
            for k, v in node.items():
                found = walk(v, path + (k,))
                if found is not None:
                    return found
        return None

    found = walk(root, ())
    if found is None:
        return None
    return found if return_path else found[0]

def merged_dict(dst: dict, src: dict, keys=None, *, overwrite: bool = True) -> dict:
    """
    Create a new dict combining dst with selected keys from src.
    If keys is None, use all keys from src.
    """
    out = dict(dst)  # shallow copy
    items = src.items() if keys is None else ((k, src[k]) for k in keys if k in src)
    if overwrite:
        out.update(items)
    else:
        for k, v in items:
            out.setdefault(k, v)
    return out


def add_index_field(arr: List[Any], *, field: str = "index") -> List[Any]:
    """
    Return a NEW list where each element is augmented with an integer index.
    - If an element is a dict, add arr_idx under `field`.
    - If not a dict, wrap it as {"value": elem, field: i}.
    """
    out: List[Any] = []
    for i, item in enumerate(arr):
        if isinstance(item, dict):
            # shallow copy to avoid mutating the input
            new_item = dict(item)
            new_item[field] = i
        else:
            new_item = {"value": item, field: i}
        out.append(new_item)
    return out

def sample_splits_json_with_index(
    data_or_str: Union[str, List[Any]],
    x_train: int,
    y_valid: int,
    z_test: int,
    *,
    seed: int | None = None,
    index_field: str = "index"
) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    Load a JSON array (or list), add a stable index to each element, then
    return X/Y/Z samples WITHOUT overlap. Each returned item contains `index_field`.
    """
    # Load if needed
    if isinstance(data_or_str, str):
        try:
            data = json.loads(data_or_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}") from e
    elif isinstance(data_or_str, list):
        data = data_or_str
    else:
        raise TypeError("data_or_str must be a JSON string or a Python list.")

    if not isinstance(data, list):
        raise ValueError("Top-level JSON must be an array (list).")

    # Add index to every element (non-destructive)
    data_idx = add_index_field(data, field=index_field)

    n = len(data_idx)
    need = x_train + y_valid + z_test
    if need > n:
        raise ValueError(f"Requested {need} samples but only {n} available.")

    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)

    train_idx = indices[:x_train]
    valid_idx = indices[x_train:x_train + y_valid]
    test_idx  = indices[x_train + y_valid:x_train + y_valid + z_test]

    train = [data_idx[i] for i in train_idx]
    valid = [data_idx[i] for i in valid_idx]
    test  = [data_idx[i] for i in test_idx]
    return train, valid, test

import json
from typing import Any, Dict

def load_dict_from_file(path: str) -> Dict[str, Any]:
    """
    Load a JSON *object* (dict) from a text file.
    Raises if the file doesn't exist, JSON is invalid, or top-level isn't a dict.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {path}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from e

    if not isinstance(obj, dict):
        raise TypeError(f"Expected a JSON object at top level in {path}, got {type(obj).__name__}")
    return obj

def load_list_from_file(path: str) -> List[Any]:
    """
    Load a JSON *array* (list) from a text file.
    Raises if the file doesn't exist, JSON is invalid, or top-level isn't a list.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {path}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from e

    if not isinstance(obj, list):
        raise TypeError(f"Expected a JSON array at top level in {path}, got {type(obj).__name__}")
    return obj


def save_dict_to_file(d: Dict[str, Any], path: str, *, pretty: bool = True) -> None:
    """
    Save a Python dict as JSON to `path`.
    - pretty=True  -> indented, human-readable
    - pretty=False -> compact
    """
    if not isinstance(d, dict):
        raise TypeError("`d` must be a dict.")
    kwargs = {"ensure_ascii": False}
    if pretty:
        kwargs["indent"] = 2
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, **kwargs)
        if pretty:
            f.write("\n")

# import json
# from typing import Any, List
def save_list_to_file(arr: List[Any], path: str, *, pretty: bool = True) -> None:
    """
    Save a Python list (JSON array) to `path`.
    - pretty=True  -> indented, human-readable
    - pretty=False -> compact
    """
    if not isinstance(arr, list):
        raise TypeError("`arr` must be a list.")
    kwargs = {"ensure_ascii": False}
    if pretty:
        kwargs["indent"] = 2
    with open(path, "w", encoding="utf-8") as f:
        json.dump(arr, f, **kwargs)
        if pretty:
            f.write("\n")

