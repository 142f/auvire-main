import os
import json
import tempfile
from functools import reduce
import operator


def keys_exists(element, keys):
    """
    Check if *keys (nested) exists in `element` (dict).
    """
    if not isinstance(element, dict):
        raise AttributeError("keys_exists() expects dict as first argument.")
    if len(keys) == 0:
        raise AttributeError("keys_exists() expects at least two arguments, one given.")

    _element = element
    for key in keys:
        try:
            _element = _element[key]
        except KeyError:
            return False
    return True


def getFromDict(dataDict, mapList):
    return reduce(operator.getitem, mapList, dataDict)


def setInDict(dataDict, mapList, value):
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value


class Logger:
    def __init__(self, folder, filename, enable=False):
        self.enable = enable
        self.path = os.path.join(folder, f"{filename}.json")
        self._document = None
        if self.enable:
            os.makedirs(folder, exist_ok=True)

    def _load(self):
        if self._document is None:
            if os.path.exists(self.path):
                with open(self.path, "r") as handle:
                    self._document = json.load(handle)
            else:
                self._document = {}
        return self._document

    def flush(self, pretty=False):
        if not self.enable or self._document is None:
            return
        folder = os.path.dirname(self.path) or "."
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=folder, delete=False, suffix=".tmp"
            ) as handle:
                temporary_path = handle.name
                json.dump(
                    self._document,
                    handle,
                    ensure_ascii=False,
                    indent=2 if pretty else None,
                    separators=None if pretty else (",", ":"),
                )
                handle.flush()
            os.replace(temporary_path, self.path)
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.remove(temporary_path)

    def create(self):
        if self.enable:
            self._document = {}
            self.flush()

    def update(self, key, content, flush=True):
        if self.enable:
            json_file = self._load()

            if isinstance(key, str):
                json_file[key] = content
            elif hasattr(key, "__iter__"):
                for i in range(len(key) - 1):
                    if not keys_exists(json_file, key[: i + 1]):
                        setInDict(json_file, key[: i + 1], {})
                setInDict(json_file, key, content)
            else:
                raise Exception("key must be either str or iterable.")

            if flush:
                self.flush()

    def get_values(self, key):
        if self.enable:
            return self._load()[key]
        else:
            return []
