import os
import hashlib
from tqdm import tqdm


def read_from_local(local_path, chunk_size=1024):
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"File '{local_path}' not found.")

    total_size = os.path.getsize(local_path)
    with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
        with open(local_path, "rb") as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                yield data
                pbar.update(len(data))


URL_MAP = {
    "vgg_lpips": "/data/gooday/taming-transformers-master/taming/vgg.pth"  # Replace with the path where your model is stored
}

CKPT_MAP = {
    "vgg_lpips": "vgg.pth"
}

MD5_MAP = {
    "vgg_lpips": "d507d7349b931f0638a25a48a722f98a"
}


def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_ckpt_path(name, root, check=False):
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        with open(path, "wb") as f:
            for data_chunk in read_from_local(URL_MAP[name]):
                f.write(data_chunk)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path


class KeyNotFoundError(Exception):
    def __init__(self, cause, keys=None, visited=None):
        self.cause = cause
        self.keys = keys
        self.visited = visited
        messages = list()
        if keys is not None:
            messages.append("Key not found: {}".format(keys))
        if visited is not None:
            messages.append("Visited: {}".format(visited))
        messages.append("Cause:\n{}".format(cause))
        message = "\n".join(messages)
        super().__init__(message)


def retrieve(
        list_or_dict, key, splitval="/", default=None, expand=True, pass_success=False
):
    keys = key.split(splitval)
    success = True
    try:
        visited = []
        parent = None
        last_key = None
        for key in keys:
            if callable(list_or_dict):
                if not expand:
                    raise KeyNotFoundError(
                        ValueError(
                            "Trying to get past callable node with expand=False."
                        ),
                        keys=keys,
                        visited=visited,
                    )
                list_or_dict = list_or_dict()
                parent[last_key] = list_or_dict

            last_key = key
            parent = list_or_dict

            try:
                if isinstance(list_or_dict, dict):
                    list_or_dict = list_or_dict[key]
                else:
                    list_or_dict = list_or_dict[int(key)]
            except (KeyError, IndexError, ValueError) as e:
                raise KeyNotFoundError(e, keys=keys, visited=visited)

            visited += [key]
        # final expansion of retrieved value
        if expand and callable(list_or_dict):
            list_or_dict = list_or_dict()
            parent[last_key] = list_or_dict
    except KeyNotFoundError as e:
        if default is None:
            raise e
        else:
            list_or_dict = default
            success = False

    if not pass_success:
        return list_or_dict
    else:
        return list_or_dict, success


if __name__ == "__main__":
    config = {"keya": "a",
              "keyb": "b",
              "keyc":
                  {"cc1": 1,
                   "cc2": 2,
                   }
              }
    print(config)
    retrieve(config, "keya")


# 加载本地模型
def load_model(name):
    model_path = CKPT_MAP.get(name)
    if model_path is None:
        raise FileNotFoundError(f"Model '{name}' not found.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    # You may add further loading logic here if required
    return model_path


# Example usage:
model_name = "vgg_lpips"
try:
    model_path = get_ckpt_path(model_name, ".")
    print(f"Model '{model_name}' loaded from '{model_path}'.")
except FileNotFoundError as e:
    print(e)
