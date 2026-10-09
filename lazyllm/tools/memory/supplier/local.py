import os
import re
import json
import time
import math
import inspect
import hashlib
import sqlite3
import threading
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple, Set

from filelock import SoftFileLock

from lazyllm import LOG, config
from ..base import LazyLLMMemoryBase


config.add('local_memory_path', str, '', 'LOCAL_MEMORY_PATH',
           description='Default SQLite file path of the local memory supplier.')

_SCHEMA = '''
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    agent_id TEXT,
    query TEXT,
    output TEXT,
    message TEXT NOT NULL,
    embedding TEXT,
    hash TEXT NOT NULL,
    size INTEGER NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(user_id, agent_id, created_at);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
'''

_TOKEN_RE = re.compile(r'[a-z0-9]+|[\u4e00-\u9fff]')
_RRF_K = 60
_RETENTION_REASONS = ('days', 'records', 'bytes', 'end_time', 'corrupt')


def _tokenize(text: Optional[str]) -> List[str]:
    return _TOKEN_RE.findall((text or '').lower())


def _cosine(vec_a: List[float], vec_b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(x * x for x in vec_a))
    norm_b = math.sqrt(sum(y * y for y in vec_b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _to_timestamp(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).timestamp()
        except ValueError:
            return float(value)
    raise TypeError(f'Unsupported end_time type: {type(value)}')


class LocalMemory():
    def __new__(cls, data_path: Optional[str] = None, *, topk: int = 10, embed=None,
                max_days: Optional[float] = None, max_records: Optional[int] = None,
                max_bytes: Optional[int] = None, end_time=None,
                auto_cleanup: bool = True) -> LazyLLMMemoryBase:
        return LocalFileMemory(
            data_path, topk=topk, embed=embed, max_days=max_days, max_records=max_records,
            max_bytes=max_bytes, end_time=end_time, auto_cleanup=auto_cleanup)


class LocalFileMemory(LazyLLMMemoryBase):
    def __init__(self, data_path: Optional[str] = None, *, topk: int = 10, embed=None,
                 max_days: Optional[float] = None, max_records: Optional[int] = None,
                 max_bytes: Optional[int] = None, end_time=None, auto_cleanup: bool = True):
        super().__init__(topk=topk)
        if not data_path:
            data_path = config['local_memory_path'] or os.path.join(config['home'], 'local_memory.db')
        self._db_path = os.path.expanduser(str(data_path))
        self._embed = embed
        self._max_days = max_days
        self._max_records = max_records
        self._max_bytes = max_bytes
        self._end_time = _to_timestamp(end_time)
        self._auto_cleanup = auto_cleanup
        self._lock = threading.RLock()
        self._active_lock = threading.Lock()
        self._active_writes: Set[Tuple] = set()
        self._ready = False
        lock_kwargs = {}
        if 'is_singleton' in inspect.signature(SoftFileLock).parameters:
            lock_kwargs['is_singleton'] = True
        self._file_lock = SoftFileLock(self._db_path + '.lock', **lock_kwargs)
        self._try_init()

    @property
    def db_path(self) -> str:
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path, timeout=10.0, isolation_level=None)

    def _try_init(self) -> bool:
        try:
            os.makedirs(os.path.dirname(self._db_path) or '.', exist_ok=True)
            try:
                conn = self._connect()
                try:
                    conn.execute('PRAGMA journal_mode = WAL;')
                    conn.execute('PRAGMA busy_timeout = 10000;')
                    conn.executescript(_SCHEMA)
                finally:
                    conn.close()
            except sqlite3.DatabaseError:
                LOG.warning(f'[LocalMemory] Corrupt database detected at {self._db_path}, quarantining it.')
                self._quarantine()
                conn = self._connect()
                try:
                    conn.execute('PRAGMA journal_mode = WAL;')
                    conn.execute('PRAGMA busy_timeout = 10000;')
                    conn.executescript(_SCHEMA)
                finally:
                    conn.close()
            self._ready = True
        except Exception as e:
            LOG.error(f'[LocalMemory] Failed to initialize local store at {self._db_path}: {e}')
            self._ready = False
        return self._ready

    def _quarantine(self):
        stamp = int(time.time())
        for suffix in ('', '-wal', '-shm'):
            path = self._db_path + suffix
            if os.path.exists(path):
                os.replace(path, f'{path}.corrupt-{stamp}')

    def _has_policy(self) -> bool:
        return (self._max_days is not None or self._max_records is not None
                or self._max_bytes is not None or self._end_time is not None)

    @staticmethod
    def _render(message: List[Dict[str, Any]]) -> str:
        return '\n'.join([f'{m.get("role", "user")}: {m.get("content", "")}' for m in message])

    @staticmethod
    def _split_message(message: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
        query = next((m.get('content') for m in reversed(message) if m.get('role') == 'user'), None)
        output = message[-1].get('content') if message and message[-1].get('role') == 'assistant' else None
        return query, output

    def _make_embedding(self, text: str) -> Optional[str]:
        if self._embed is None:
            return None
        try:
            vec = [float(x) for x in self._embed(text)]
            return json.dumps(vec, ensure_ascii=False) if vec else None
        except Exception as e:
            LOG.warning(f'[LocalMemory] Embedding adapter failed, fallback to keyword matching: {e}')
            return None

    def _add(self, message: List[Dict[str, Any]], user_id: Optional[str] = None,
             agent_id: Optional[str] = None):
        if not self._ready and not self._try_init():
            return None
        query, output = self._split_message(message)
        payload = json.dumps(message, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(payload.encode('utf-8')).hexdigest()
        size = (len(payload.encode('utf-8')) + len((query or '').encode('utf-8'))
                + len((output or '').encode('utf-8')))
        embedding = self._make_embedding(self._render(message))
        token = (user_id, agent_id, digest)
        try:
            with self._active_lock:
                self._active_writes.add(token)
            try:
                with self._lock, self._file_lock:
                    conn = self._connect()
                    try:
                        conn.execute(
                            'INSERT INTO memories (user_id, agent_id, query, output, message, embedding, '
                            'hash, size, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                            (user_id, agent_id, query, output, payload, embedding, digest,
                             size, time.time()))
                    finally:
                        conn.close()
            finally:
                with self._active_lock:
                    self._active_writes.discard(token)
            if self._auto_cleanup and self._has_policy():
                self.cleanup()
        except Exception as e:
            LOG.error(f'[LocalMemory] Failed to add memory: {e}')

    def _fetch_rows(self, user_id: Optional[str], agent_id: Optional[str]) -> List[sqlite3.Row]:
        sql, params = 'SELECT * FROM memories', []
        if user_id is not None:
            sql += ' WHERE user_id = ?'
            params.append(user_id)
            if agent_id is not None:
                sql += ' AND agent_id = ?'
                params.append(agent_id)
        elif agent_id is not None:
            sql += ' WHERE agent_id = ?'
            params.append(agent_id)
        sql += ' ORDER BY created_at DESC, id DESC'
        conn = self._connect()
        try:
            conn.row_factory = sqlite3.Row
            return conn.execute(sql, params).fetchall()
        finally:
            conn.close()

    def _get(self, query: Optional[str] = None, user_id: Optional[str] = None,
             agent_id: Optional[str] = None) -> str:
        if not self._ready and not self._try_init():
            return ''
        try:
            try:
                rows = self._fetch_rows(user_id, agent_id)
            except sqlite3.DatabaseError:
                if not self._try_init():
                    return ''
                rows = self._fetch_rows(user_id, agent_id)
            records = []
            for row in rows:
                try:
                    records.append((row, json.loads(row['message'])))
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
            if not records:
                return ''
            if query:
                ranked = self._rank(query, records)
            else:
                ranked = sorted(records, key=lambda item: item[0]['created_at'], reverse=True)
            return '\n'.join([self._render(message) for _, message in ranked[:self._topk]])
        except Exception as e:
            LOG.error(f'[LocalMemory] Failed to get memories: {e}')
            return ''

    def _keyword_rank(self, query: str, records: List[Tuple[sqlite3.Row, list]]) -> List[Tuple]:
        qtokens = set(_tokenize(query))
        scored = []
        for row, message in records:
            tokens = _tokenize(self._render(message))
            if not tokens:
                continue
            tf = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            hits = sum(min(tf.get(token, 0), 3) for token in qtokens)
            if hits <= 0:
                continue
            scored.append((hits / (1.0 + math.log1p(len(tokens))), row['created_at'], row, message))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [(row, message) for _, _, row, message in scored]

    def _embedding_rank(self, qvec: List[float], records: List[Tuple[sqlite3.Row, list]]) -> List[Tuple]:
        scored = []
        for row, message in records:
            if not row['embedding']:
                continue
            try:
                vec = [float(x) for x in json.loads(row['embedding'])]
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if len(vec) != len(qvec):
                continue
            score = _cosine(qvec, vec)
            if score > 0:
                scored.append((score, row['created_at'], row, message))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [(row, message) for _, _, row, message in scored]

    @staticmethod
    def _fuse(rank_a: List[Tuple], rank_b: List[Tuple]) -> List[Tuple]:
        scores, order = {}, {}
        for rank in (rank_a, rank_b):
            for idx, item in enumerate(rank):
                key = item[0]['id']
                scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + idx + 1)
                order.setdefault(key, item)
        return [order[key] for key in sorted(
            scores, key=lambda key: (scores[key], order[key][0]['created_at']), reverse=True)]

    def _rank(self, query: str, records: List[Tuple[sqlite3.Row, list]]) -> List[Tuple]:
        keyword_rank = self._keyword_rank(query, records)
        qvec = None
        if self._embed is not None:
            try:
                qvec = [float(x) for x in self._embed(query)]
            except Exception as e:
                LOG.warning(f'[LocalMemory] Embedding adapter failed on query, use keyword matching: {e}')
        if qvec:
            return self._fuse(keyword_rank, self._embedding_rank(qvec, records))
        return keyword_rank

    def cleanup(self, dry_run: bool = False, now: Optional[float] = None) -> Dict[str, Any]:
        result = {'dry_run': dry_run, 'deleted': 0, 'bytes': 0, 'remaining': 0,
                  'candidates': [], 'reasons': {key: 0 for key in _RETENTION_REASONS}}
        if not self._ready and not self._try_init():
            return result
        now = time.time() if now is None else now
        try:
            with self._lock, self._file_lock:
                conn = self._connect()
                try:
                    conn.row_factory = sqlite3.Row
                    return self._cleanup_conn(conn, dry_run, now, result)
                finally:
                    conn.close()
        except Exception as e:
            LOG.error(f'[LocalMemory] Failed to cleanup memories: {e}')
            return result

    def _is_corrupt(self, row: sqlite3.Row) -> bool:
        try:
            json.loads(row['message'])
            return False
        except (TypeError, ValueError, json.JSONDecodeError):
            return True

    def _expiry_reason(self, row: sqlite3.Row, now: float) -> Optional[str]:
        if self._max_days is not None and row['created_at'] < now - self._max_days * 86400:
            return 'days'
        if self._end_time is not None and row['created_at'] < self._end_time:
            return 'end_time'
        return None

    def _row_status(self, row: sqlite3.Row, active: Set[Tuple], now: float) -> str:
        token = (row['user_id'], row['agent_id'], row['hash'])
        if token in active:
            return 'active'
        if self._is_corrupt(row):
            return 'corrupt'
        return self._expiry_reason(row, now) or 'survivor'

    def _apply_records_limit(self, marked: Dict[int, Tuple[str, int]], survivors: List[sqlite3.Row],
                             alive_size: int) -> int:
        if self._max_records is None or len(survivors) <= self._max_records:
            return alive_size
        for row in survivors[:len(survivors) - self._max_records]:
            marked[row['id']] = ('records', row['size'])
            alive_size -= row['size']
        return alive_size

    def _apply_bytes_limit(self, marked: Dict[int, Tuple[str, int]], survivors: List[sqlite3.Row],
                           alive_size: int):
        if self._max_bytes is None or alive_size <= self._max_bytes:
            return
        for row in survivors:
            if alive_size <= self._max_bytes:
                break
            if row['id'] not in marked:
                marked[row['id']] = ('bytes', row['size'])
                alive_size -= row['size']

    def _finish_cleanup(self, conn: sqlite3.Connection, rows: List[sqlite3.Row],
                        marked: Dict[int, Tuple[str, int]], dry_run: bool,
                        result: Dict[str, Any]) -> Dict[str, Any]:
        ids = sorted(marked.keys())
        result['candidates'] = ids
        result['deleted'] = len(ids)
        result['bytes'] = sum(item[1] for item in marked.values())
        result['remaining'] = len(rows) - (0 if dry_run else len(ids))
        for reason, _ in marked.values():
            result['reasons'][reason] += 1
        if not dry_run and ids:
            conn.execute('BEGIN IMMEDIATE')
            try:
                conn.executemany('DELETE FROM memories WHERE id = ?', [(idx,) for idx in ids])
                conn.execute('COMMIT')
            except Exception:
                conn.execute('ROLLBACK')
                raise
        return result

    def _cleanup_conn(self, conn: sqlite3.Connection, dry_run: bool, now: float,
                      result: Dict[str, Any]) -> Dict[str, Any]:
        rows = conn.execute(
            'SELECT id, user_id, agent_id, hash, size, created_at, message FROM memories '
            'ORDER BY created_at ASC, id ASC').fetchall()
        with self._active_lock:
            active = set(self._active_writes)
        marked: Dict[int, Tuple[str, int]] = {}
        survivors: List[sqlite3.Row] = []
        alive_size = 0
        for row in rows:
            status = self._row_status(row, active, now)
            if status == 'active':
                continue
            if status == 'survivor':
                survivors.append(row)
                alive_size += row['size']
            else:
                marked[row['id']] = (status, row['size'])
        alive_size = self._apply_records_limit(marked, survivors, alive_size)
        self._apply_bytes_limit(marked, survivors, alive_size)
        return self._finish_cleanup(conn, rows, marked, dry_run, result)
