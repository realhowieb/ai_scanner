"""Stand-in for the `streamlit` module, now that the Streamlit app is gone (2026-10-21).

Shared helpers written for the old app still say `st.cache_data`, `st.session_state`
and `st.secrets`; the API and the scheduled jobs call them. This keeps those working
without loading Streamlit (about 35 MB in the API, plus plotly and pyarrow):

- `st.cache_data(ttl=..., max_entries=...)` is a real in-process cache that behaves like
  Streamlit's outside a Streamlit app: parameters whose names start with "_" are left
  out of the key, every caller gets its own deep copy of the value (as Streamlit's
  pickled copies did, so mutating a cached DataFrame can't change the cache), and `fn.clear()` /
  `st.cache_data.clear()` empty it. `st.cache_resource` shares the object instead.
- `st.session_state` is one process-wide dict, as Streamlit's is without a running app.
- `st.secrets` is empty; settings come from the environment.
- Anything else (`st.markdown`, `st.columns`, ...) is a silent no-op, for render code that
  no live path reaches.
"""
from __future__ import annotations

import copy
import functools
import inspect
import pickle
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Hashable, List, Optional, Tuple

_MISSING = object()


def _ttl_seconds(ttl: Any) -> Optional[float]:
    if ttl is None:
        return None
    if isinstance(ttl, (int, float)):
        return float(ttl)
    if hasattr(ttl, "total_seconds"):
        return float(ttl.total_seconds())
    text = str(ttl).strip().lower()
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    try:
        if text[-1:] in units:
            return float(text[:-1]) * units[text[-1]]
        return float(text)
    except ValueError:
        return None


def _freeze(value: Any) -> Hashable:
    """A hashable stand-in for an argument (lists, dicts, sets, DataFrames)."""
    try:
        hash(value)
        return (type(value).__name__, value)
    except TypeError:
        pass
    if isinstance(value, dict):
        return ("dict", tuple(sorted(((repr(k), _freeze(v)) for k, v in value.items()))))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_freeze(v) for v in value))
    if isinstance(value, (set, frozenset)):
        return ("set", tuple(sorted(repr(v) for v in value)))
    try:
        return ("pickle", pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:
        return ("repr", repr(value))


class _Cached:
    """One cached function. Thread-safe; a miss runs the function outside the lock."""

    def __init__(self, fn: Callable[..., Any], ttl: Optional[float], max_entries: Optional[int],
                 copy_values: bool) -> None:
        self._fn = fn
        self._ttl = ttl
        self._max = max_entries if max_entries and max_entries > 0 else None
        self._copy = copy_values
        self._lock = threading.Lock()
        self._items: "OrderedDict[Hashable, Tuple[float, Any]]" = OrderedDict()
        try:
            self._sig: Optional[inspect.Signature] = inspect.signature(fn)
        except (TypeError, ValueError):
            self._sig = None
        functools.update_wrapper(self, fn)

    def _key(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Hashable:
        if self._sig is not None:
            try:
                bound = self._sig.bind(*args, **kwargs)
                bound.apply_defaults()
                return tuple((name, _freeze(v)) for name, v in bound.arguments.items() if not name.startswith("_"))
            except TypeError:
                pass
        return (_freeze(args), _freeze(kwargs))

    def _store(self, value: Any) -> Any:
        return copy.deepcopy(value) if self._copy else value

    def _load(self, stored: Any) -> Any:
        return copy.deepcopy(stored) if self._copy else stored

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        key = self._key(args, kwargs)
        now = time.monotonic()
        with self._lock:
            hit = self._items.get(key, _MISSING)
            if hit is not _MISSING:
                expires, stored = hit  # type: ignore[misc]
                if self._ttl is None or now < expires:
                    self._items.move_to_end(key)
                    return self._load(stored)
                del self._items[key]
        value = self._fn(*args, **kwargs)
        with self._lock:
            self._items[key] = (now + (self._ttl or 0.0), self._store(value))
            self._items.move_to_end(key)
            while self._max is not None and len(self._items) > self._max:
                self._items.popitem(last=False)
        return value

    def clear(self, *_args: Any, **_kwargs: Any) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        return len(self._items)


class _CacheDecorator:
    """st.cache_data / st.cache_resource: usable bare (@st.cache_data) or with options."""

    def __init__(self, copy_values: bool) -> None:
        self._copy = copy_values
        self._all: List[_Cached] = []
        self._lock = threading.Lock()

    def __call__(self, func: Optional[Callable[..., Any]] = None, *, ttl: Any = None,
                 max_entries: Optional[int] = None, **_ignored: Any) -> Any:
        def wrap(fn: Callable[..., Any]) -> _Cached:
            cached = _Cached(fn, _ttl_seconds(ttl), max_entries, self._copy)
            with self._lock:
                self._all.append(cached)
            return cached

        return wrap(func) if callable(func) else wrap

    def clear(self) -> None:
        with self._lock:
            caches = list(self._all)
        for c in caches:
            c.clear()


class _SessionState(dict):
    """Dict with attribute access, like st.session_state."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        self.pop(name, None)


class _NoOp:
    """Absorbs any Streamlit UI call: callable, a context manager, unpackable into columns."""

    def __call__(self, *args: Any, **kwargs: Any) -> "_NoOp":
        return self

    def __getattr__(self, _name: str) -> "_NoOp":
        return self

    def __enter__(self) -> "_NoOp":
        return self

    def __exit__(self, *_exc: Any) -> bool:
        return False

    def __iter__(self):
        return iter(())

    def __bool__(self) -> bool:
        return False


class _StreamlitCompat:
    def __init__(self) -> None:
        self.cache_data = _CacheDecorator(copy_values=True)
        self.cache_resource = _CacheDecorator(copy_values=False)
        self.session_state = _SessionState()
        self.secrets: Dict[str, Any] = {}
        self.query_params: Dict[str, Any] = {}
        self._noop = _NoOp()

    def columns(self, spec: Any = 1, *_a: Any, **_k: Any) -> List[_NoOp]:
        n = spec if isinstance(spec, int) else len(spec) if hasattr(spec, "__len__") else 1
        return [self._noop for _ in range(max(1, n))]

    def tabs(self, labels: Any, *_a: Any, **_k: Any) -> List[_NoOp]:
        return [self._noop for _ in labels]

    def __getattr__(self, _name: str) -> _NoOp:
        return self._noop


st = _StreamlitCompat()
