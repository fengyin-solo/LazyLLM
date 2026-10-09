import os
import sys
import time
import uuid
import zlib
import struct
import pickle
import hashlib
import inspect
import threading
import types
from contextvars import ContextVar
from os import PathLike
from asyncio import events

import cloudpickle

from ..common.bind import Bind, Placeholder
from ..common.logger import LOG

_VERSION = 1
_MAGIC = b'LZLFCK01'
_STEP_PREFIX = 's'
_PROGRESS_PREFIX = 'lp'
_STATUS_SUCCESS = 'success'
_DEFAULT_DIR = '.lazyllm_flow_checkpoints'


class CheckpointStoreBase(object):
    # Persistent key-value backend. load returns None on miss; dump must be durable
    # before it returns. Implementations only deal with opaque bytes.
    def load(self, key: str):
        raise NotImplementedError

    def dump(self, key: str, data: bytes):
        raise NotImplementedError


class FileCheckpointStore(CheckpointStoreBase):
    # One immutable file per key. Files are written to a unique temp name and then
    # atomically renamed, so concurrent processes never observe a half-written file.
    def __init__(self, root):
        self.root = os.path.abspath(os.path.expanduser(os.fspath(root)))

    def __repr__(self):
        return f'FileCheckpointStore({self.root})'

    def _path(self, key):
        return os.path.join(self.root, key[:2], f'{key}.ckpt')

    def dump(self, key, data):
        final_path = self._path(key)
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        tmp_path = os.path.join(
            os.path.dirname(final_path), f'.{key}.{os.getpid()}.{uuid.uuid4().hex}.tmp')
        try:
            with open(tmp_path, 'wb') as fp:
                fp.write(data)
                fp.flush()
                os.fsync(fp.fileno())
            os.replace(tmp_path, final_path)
            try:
                dir_fd = os.open(os.path.dirname(final_path), os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except OSError:
                pass
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass

    def load(self, key):
        try:
            with open(self._path(key), 'rb') as fp:
                return fp.read()
        except FileNotFoundError:
            return None
        except OSError:
            return None


_USER_STORE_METHODS = (('load', 'dump'), ('get', 'set'), ('read', 'write'))


class _UserStoreAdapter(CheckpointStoreBase):
    def __init__(self, obj, load_name, dump_name):
        self._obj = obj
        self._load_name = load_name
        self._dump_name = dump_name

    def load(self, key):
        return getattr(self._obj, self._load_name)(key)

    def dump(self, key, data):
        getattr(self._obj, self._dump_name)(key, data)


def normalize_checkpoint(cfg):
    if cfg is None:
        return None
    if cfg is False:
        return False
    if cfg is True or cfg == 'default':
        return 'default'
    if isinstance(cfg, CheckpointStoreBase):
        return cfg
    if isinstance(cfg, (str, PathLike)):
        return FileCheckpointStore(cfg)
    for load_name, dump_name in _USER_STORE_METHODS:
        if hasattr(cfg, load_name) and hasattr(cfg, dump_name):
            return _UserStoreAdapter(cfg, load_name, dump_name)
    raise TypeError(f'Unsupported checkpoint config: {cfg!r}')


def default_store(checkpoint_dir=None):
    root = checkpoint_dir or os.path.join(os.getcwd(), _DEFAULT_DIR)
    return FileCheckpointStore(root)


# -----------------------------------------------------------------------------
# Signatures
# -----------------------------------------------------------------------------

def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _stable_bytes(obj):
    try:
        return cloudpickle.dumps(obj)
    except Exception:
        try:
            return repr(obj).encode('utf-8', 'replace')
        except Exception:
            return b'<unhashable-object>'


def _code_parts(code):
    return (
        code.co_code,
        code.co_argcount,
        code.co_kwonlyargcount,
        tuple(_const_parts(c) for c in code.co_consts),
        tuple(code.co_names),
    )


def _const_parts(c):
    if isinstance(c, types.CodeType):
        return ('code', _code_parts(c))
    if isinstance(c, type):
        return ('type', _type_parts(c))
    if callable(c):
        return ('callable', _callable_parts(c))
    return ('v', _stable_bytes(c))


def _type_parts(t):
    methods = tuple(sorted(
        (k, _code_parts(v.__code__)) for k, v in vars(t).items()
        if isinstance(v, types.FunctionType)))
    return ('type', t.__module__, getattr(t, '__qualname__', t.__name__), methods)


def _arg_parts(a):
    if isinstance(a, Bind.Args):
        return ('ref', a._target_id,
                None if a._item_key is Bind.Args._None else a._item_key,
                None if a._attr_key is Bind.Args._None else a._attr_key,
                a._unpack)
    if isinstance(a, Placeholder):
        return ('placeholder', a.idx)
    if isinstance(a, type):
        return ('type', _type_parts(a))
    if callable(a):
        return ('callable', _callable_parts(a))
    return ('v', _stable_bytes(a))


def _func_parts(f):
    closure = None
    if f.__closure__:
        closure = tuple(_arg_parts(c.cell_contents) for c in f.__closure__)
    return ('func', f.__module__, f.__qualname__, f.__code__.co_filename,
            f.__code__.co_firstlineno, _code_parts(f.__code__),
            _stable_bytes(f.__defaults__), _stable_bytes(f.__kwdefaults__), closure)


def _callable_parts(f):
    if isinstance(f, Bind):
        return ('bind', _callable_parts(f._f),
                tuple(_arg_parts(a) for a in f._args),
                tuple(sorted((k, _arg_parts(v)) for k, v in f._kw.items())))
    try:
        from .flow import FlowBase, _FuncWrap
    except ImportError:
        FlowBase, _FuncWrap = None, None
    if FlowBase is not None and isinstance(f, FlowBase):
        return ('flow', f._checkpoint_parts(),
                tuple(_callable_parts(i) for i in f._items))
    if _FuncWrap is not None and type(f) is _FuncWrap:
        return ('funcwrap', _callable_parts(f._f))
    if isinstance(f, (types.BuiltinFunctionType, types.BuiltinMethodType)):
        return ('builtin', getattr(f, '__module__', None),
                getattr(f, '__qualname__', repr(f)))
    if isinstance(f, type):
        return _type_parts(f)
    if inspect.isfunction(f):
        return _func_parts(f)
    if inspect.ismethod(f):
        self_obj = f.__self__
        return ('method', _func_parts(f.__func__),
                None if inspect.ismodule(self_obj) else _stable_bytes(self_obj))
    if callable(f):
        cls = type(f)
        state = tuple(sorted(
            (k, _stable_bytes(v)) for k, v in vars(f).items() if not k.startswith('_')))
        return ('obj', _type_parts(cls), getattr(f, '_module_id', None), state)
    return ('v', _stable_bytes(f))


def function_signature(f):
    # Hex signature of the callable's code and configuration. Returns None when the
    # signature cannot be built, in which case the step is simply not cached.
    if f is None:
        return None
    try:
        return _sha(pickle.dumps(_callable_parts(f), protocol=4))
    except Exception as e:
        LOG.debug(f'Build function signature failed for {f!r}: {e}')
        return None


def value_signature(v):
    try:
        return _sha(cloudpickle.dumps(v))
    except Exception as e:
        LOG.debug(f'Build value signature failed for {type(v)}: {e}')
        return None


def chain_signature(prev, sig):
    return _sha(prev.encode('ascii') + b'|' + sig.encode('ascii'))


# -----------------------------------------------------------------------------
# Keys and records
# -----------------------------------------------------------------------------

def _hash_key(kind, parts):
    data = pickle.dumps(('lazyllm-flow-checkpoint', _VERSION, kind, parts), protocol=4)
    return f'{kind}_' + _sha(data)


def build_step_key(root_name, flow_path, config_sig, prefix_sig,
                   step_id, iterations, input_sig):
    if config_sig is None or prefix_sig is None or input_sig is None:
        return None
    return _hash_key(_STEP_PREFIX, (root_name, tuple(flow_path), config_sig, prefix_sig,
                                    tuple(step_id), tuple(iterations), input_sig))


def build_loop_progress_key(root_name, flow_path, config_sig, iteration, input_sig):
    if config_sig is None or input_sig is None:
        return None
    return _hash_key(_PROGRESS_PREFIX, (root_name, tuple(flow_path), config_sig,
                                        int(iteration), input_sig))


def _encode(key, record):
    record = dict(record)
    record['key'] = key
    record['version'] = _VERSION
    payload = cloudpickle.dumps(record)
    crc = zlib.crc32(payload) & 0xffffffff
    return _MAGIC + struct.pack('>Q', len(payload)) + payload + struct.pack('>I', crc)


def _decode(data, key):
    try:
        if not isinstance(data, (bytes, bytearray)) or not data.startswith(_MAGIC):
            return None
        offset = len(_MAGIC)
        if len(data) < offset + 8 + 4:
            return None
        length = struct.unpack('>Q', bytes(data[offset:offset + 8]))[0]
        payload_start = offset + 8
        payload_end = payload_start + length
        if len(data) != payload_end + 4:
            return None
        payload = bytes(data[payload_start:payload_end])
        crc = struct.unpack('>I', bytes(data[payload_end:payload_end + 4]))[0]
        if (zlib.crc32(payload) & 0xffffffff) != crc:
            return None
        record = cloudpickle.loads(payload)
        if not isinstance(record, dict) or record.get('key') != key:
            return None
        if record.get('version') != _VERSION or record.get('status') != _STATUS_SUCCESS:
            return None
        return record
    except Exception as e:
        LOG.debug(f'Ignore unreadable checkpoint {key}: {e}')
        return None


def load_output(store, key):
    # Never raises: a corrupt/incomplete checkpoint is treated as a cache miss.
    try:
        data = store.load(key)
    except Exception as e:
        LOG.warning(f'Checkpoint store load failed for {key}: {e}')
        return None
    if data is None:
        return None
    record = _decode(data, key)
    return record


def dump_output(store, key, output, meta=None):
    # Best effort: a checkpoint write failure never masks the business result.
    record = {'status': _STATUS_SUCCESS, 'output': output,
              'meta': meta or {}, 'created_at': time.time()}
    try:
        store.dump(key, _encode(key, record))
    except Exception as e:
        LOG.warning(f'Checkpoint store dump failed for {key}: {e}')


def load_loop_progress(store, key):
    record = load_output(store, key)
    if record is None:
        return None
    progress = record.get('meta', {}).get('loop_progress')
    if not isinstance(progress, dict) or 'output' not in progress:
        return None
    if not isinstance(progress.get('stopped'), bool):
        return None
    if not isinstance(progress.get('runtime_count'), int):
        return None
    return progress


def dump_loop_progress(store, key, output, stopped, runtime_count):
    progress = {'output': output, 'stopped': bool(stopped),
                'runtime_count': int(runtime_count)}
    dump_output(store, key, output, meta={'loop_progress': progress,
                                          'iteration_kind': 'completed'})


# -----------------------------------------------------------------------------
# Loop iteration frames
# -----------------------------------------------------------------------------

# Each frame is (flow_path_tuple, iteration_index). The value is an immutable tuple
# so copy_context() snapshots in worker threads cannot leak pushes back to the
# parent flow, and spawned processes receive a plain picklable snapshot.
_iter_frames_var = ContextVar('lazyllm.flow.checkpoint_iter_frames', default=())
_iter_frames_tl = threading.local()


def _frames_async():
    return events._get_running_loop() is not None


def current_frames():
    if _frames_async():
        return _iter_frames_var.get()
    return getattr(_iter_frames_tl, 'frames', ())


def push_iteration(flow_path, iteration):
    frame = (tuple(flow_path), int(iteration))
    if _frames_async():
        return ('var', _iter_frames_var.set(current_frames() + (frame,)))
    old = getattr(_iter_frames_tl, 'frames', ())
    _iter_frames_tl.frames = old + (frame,)
    return ('tl', old)


def pop_iteration(token):
    kind, state = token
    if kind == 'var':
        _iter_frames_var.reset(state)
    else:
        _iter_frames_tl.frames = state


def snapshot_frames():
    try:
        return tuple(current_frames())
    except Exception:
        return ()


def install_frames(frames):
    frames = tuple(frames)
    _iter_frames_tl.frames = frames
    _iter_frames_var.set(frames)


def enclosing_loop_iterations(flow_path, frames):
    # Keep frames of Loop flows that are the current flow itself or its ancestors.
    flow_path = tuple(flow_path)
    result = []
    for frame_path, iteration in frames:
        frame_path = tuple(frame_path)
        if flow_path == frame_path or (
                len(flow_path) > len(frame_path) and flow_path[:len(frame_path)] == frame_path):
            result.append((frame_path, iteration))
    return tuple(result)


def config_flag_enabled(store_or_flag):
    return store_or_flag not in (None, False)
