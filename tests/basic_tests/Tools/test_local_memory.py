import os
import json
import time
import sqlite3
import threading

import pytest

import lazyllm
from lazyllm.tools.memory import Memory, memory_hook
from lazyllm.tools.memory.supplier.local import LocalMemory, LocalFileMemory


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / 'local_memory.db')


def _open(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _rows(db_path):
    with _open(db_path) as conn:
        return conn.execute('SELECT * FROM memories ORDER BY id ASC').fetchall()


class TestLocalMemoryAddGet:
    def test_factory_source_local(self, db_path):
        m = Memory(source='local', data_path=db_path)
        assert isinstance(m, LocalFileMemory)

    def test_add_get_roundtrip(self, db_path):
        m = LocalFileMemory(db_path, topk=5)
        m.add('My Nova 2000 order arrived damaged', 'We are sorry to hear that',
              user_id='u1', agent_id='a1')
        result = m.get('Nova damaged order', user_id='u1', agent_id='a1')
        assert 'Nova 2000 order arrived damaged' in result
        assert 'We are sorry to hear that' in result
        assert result.startswith('user:')
        assert 'assistant:' in result

    def test_add_persists_all_columns(self, db_path):
        m = LocalFileMemory(db_path)
        before = time.time()
        m.add('a question about rockets', 'rockets fly fast', user_id='u1', agent_id='a1')
        after = time.time()
        rows = _rows(db_path)
        assert len(rows) == 1
        row = dict(rows[0])
        assert row['user_id'] == 'u1'
        assert row['agent_id'] == 'a1'
        assert row['query'] == 'a question about rockets'
        assert row['output'] == 'rockets fly fast'
        messages = json.loads(row['message'])
        assert messages == [{'role': 'user', 'content': 'a question about rockets'},
                            {'role': 'assistant', 'content': 'rockets fly fast'}]
        assert len(row['hash']) == 64
        assert row['size'] > 0
        assert row['embedding'] is None
        assert before <= row['created_at'] <= after

    def test_add_with_history_is_normalized(self, db_path):
        m = LocalFileMemory(db_path)
        m.add('another question', 'another answer', history=[['old question', 'old answer']],
              user_id='u1', agent_id='a1')
        row = dict(_rows(db_path)[0])
        messages = json.loads(row['message'])
        assert [msg['content'] for msg in messages] == [
            'old question', 'old answer', 'another question', 'another answer']
        assert row['query'] == 'another question'
        assert row['output'] == 'another answer'

    def test_keyword_retrieval_english_and_chinese(self, db_path):
        m = LocalFileMemory(db_path, topk=5)
        m.add('my flight ticket was cancelled', 'please contact support', user_id='u1', agent_id='a1')
        m.add('today weather is sunny', 'great for hiking', user_id='u1', agent_id='a1')
        m.add('我的航班行李延误了', '我们正在查询', user_id='u2', agent_id='a1')
        flight = m.get('cancelled flight ticket', user_id='u1', agent_id='a1')
        assert 'flight ticket was cancelled' in flight
        assert 'weather' not in flight
        chinese = m.get('行李 航班', user_id='u2', agent_id='a1')
        assert '行李延误' in chinese

    def test_topk_is_respected(self, db_path):
        m = LocalFileMemory(db_path, topk=2)
        for idx in range(5):
            m.add(f'shared keyword question {idx}', f'answer {idx}', user_id='u1', agent_id='a1')
        result = m.get('shared keyword', user_id='u1', agent_id='a1')
        assert len([line for line in result.splitlines() if line.startswith('user:')]) == 2

    def test_get_without_query_returns_latest(self, db_path):
        m = LocalFileMemory(db_path, topk=10)
        for idx in range(3):
            m.add(f'question number {idx}', f'answer {idx}', user_id='u1', agent_id='a1')
        result = m.get(user_id='u1', agent_id='a1')
        lines = [line for line in result.splitlines() if line.startswith('user:')]
        assert lines[0] == 'user: question number 2'

    def test_scoping_by_user_and_agent(self, db_path):
        m = LocalFileMemory(db_path)
        m.add('secret alpha question', 'alpha answer', user_id='u1', agent_id='a1')
        m.add('secret beta question', 'beta answer', user_id='u2', agent_id='a2')
        assert 'alpha' in m.get('secret', user_id='u1', agent_id='a1')
        assert 'beta' not in m.get('secret', user_id='u1', agent_id='a1')
        assert m.get('secret', user_id='unknown') == ''

    def test_no_match_returns_empty_string(self, db_path):
        m = LocalFileMemory(db_path)
        m.add('something entirely different', 'indeed', user_id='u1', agent_id='a1')
        assert m.get('nonexistent xyzzy', user_id='u1', agent_id='a1') == ''

    def test_content_hash_dedupes_equivalent_payload(self, db_path):
        m = LocalFileMemory(db_path)
        m.add('same question', 'same answer', user_id='u1', agent_id='a1')
        m.add('same question', 'same answer', user_id='u1', agent_id='a1')
        rows = _rows(db_path)
        assert rows[0]['hash'] == rows[1]['hash']


class TestLocalMemoryPersistence:
    def test_data_survives_new_instance(self, db_path):
        m1 = LocalFileMemory(db_path, topk=5)
        m1.add('persisted question about planets', 'saturn has rings', user_id='u1', agent_id='a1')
        m2 = LocalFileMemory(db_path, topk=5)
        result = m2.get('planets', user_id='u1', agent_id='a1')
        assert 'saturn has rings' in result
        assert len(_rows(db_path)) == 1

    def test_default_path_from_config(self, tmp_path):
        path = str(tmp_path / 'cfg_memory.db')
        with lazyllm.config.temp('local_memory_path', path):
            m = LocalMemory()
            assert m.db_path == path
            m.add('config path question', 'config path answer', user_id='u1', agent_id='a1')
        assert os.path.exists(path)
        assert len(_rows(path)) == 1


class TestLocalMemoryRetention:
    def _backdate(self, db_path, query, days):
        with _open(db_path) as conn:
            conn.execute('UPDATE memories SET created_at = ? WHERE query = ?',
                         (time.time() - days * 86400, query))

    def test_max_days(self, db_path):
        m = LocalFileMemory(db_path, max_days=7, auto_cleanup=False)
        m.add('fresh question', 'fresh answer', user_id='u1', agent_id='a1')
        m.add('ancient question', 'ancient answer', user_id='u1', agent_id='a1')
        self._backdate(db_path, 'ancient question', 30)
        report = m.cleanup()
        assert report['deleted'] == 1
        assert report['reasons']['days'] == 1
        remaining = [row['query'] for row in _rows(db_path)]
        assert remaining == ['fresh question']

    def test_max_records(self, db_path):
        m = LocalFileMemory(db_path, max_records=3, auto_cleanup=False)
        for idx in range(7):
            m.add(f'record question {idx}', f'answer {idx}', user_id='u1', agent_id='a1')
        report = m.cleanup()
        assert report['deleted'] == 4
        assert report['reasons']['records'] == 4
        assert report['remaining'] == 3
        assert [row['query'] for row in _rows(db_path)] == [
            f'record question {idx}' for idx in range(4, 7)]

    def test_max_bytes(self, db_path):
        m = LocalFileMemory(db_path, max_bytes=220, auto_cleanup=False)
        for idx in range(6):
            m.add(f'byte limit question {idx} with some padding text',
                  f'answer {idx} padding', user_id='u1', agent_id='a1')
        report = m.cleanup()
        assert report['deleted'] >= 1
        assert report['reasons']['bytes'] == report['deleted']
        with _open(db_path) as conn:
            total = conn.execute('SELECT COALESCE(SUM(size), 0) FROM memories').fetchone()[0]
        assert total <= 220

    def test_end_time_epoch_and_iso(self, db_path):
        m = LocalFileMemory(db_path, end_time=time.time() + 100000, auto_cleanup=False)
        m.add('before cutoff question', 'answer', user_id='u1', agent_id='a1')
        report = m.cleanup()
        assert report['deleted'] == 1
        assert report['reasons']['end_time'] == 1
        assert _rows(db_path) == []

        m2 = LocalFileMemory(db_path.replace('.db', 'iso.db'), end_time='2099-01-01T00:00:00',
                             auto_cleanup=False)
        m2.add('iso cutoff question', 'answer', user_id='u1', agent_id='a1')
        assert m2.cleanup()['reasons']['end_time'] == 1

    def test_combined_dimensions(self, db_path):
        m = LocalFileMemory(db_path, max_days=7, max_records=10, end_time=time.time() - 100,
                            auto_cleanup=False)
        m.add('new question', 'new answer', user_id='u1', agent_id='a1')
        m.add('old question', 'old answer', user_id='u1', agent_id='a1')
        self._backdate(db_path, 'old question', 30)
        report = m.cleanup()
        assert report['deleted'] == 1
        assert report['reasons']['days'] == 1
        assert report['reasons']['end_time'] == 0
        assert report['remaining'] == 1

    def test_dry_run_does_not_delete(self, db_path):
        m = LocalFileMemory(db_path, max_records=1, auto_cleanup=False)
        for idx in range(4):
            m.add(f'dry question {idx}', f'answer {idx}', user_id='u1', agent_id='a1')
        report = m.cleanup(dry_run=True)
        assert report['dry_run'] is True
        assert report['deleted'] == 3
        assert len(report['candidates']) == 3
        assert len(_rows(db_path)) == 4
        real = m.cleanup(dry_run=False)
        assert real['deleted'] == 3
        assert len(_rows(db_path)) == 1

    def test_auto_cleanup_after_add(self, db_path):
        m = LocalFileMemory(db_path, max_records=2)
        for idx in range(5):
            m.add(f'auto question {idx}', f'answer {idx}', user_id='u1', agent_id='a1')
        assert len(_rows(db_path)) == 2

    def test_cleanup_without_policy_keeps_everything(self, db_path):
        m = LocalFileMemory(db_path, auto_cleanup=False)
        for idx in range(5):
            m.add(f'keep question {idx}', f'answer {idx}', user_id='u1', agent_id='a1')
        report = m.cleanup()
        assert report['deleted'] == 0
        assert report['remaining'] == 5
        assert len(_rows(db_path)) == 5

    def test_cleanup_skips_active_agent_writes(self, db_path):
        m = LocalFileMemory(db_path, max_records=2, auto_cleanup=False)
        for idx in range(4):
            m.add(f'active question {idx}', f'answer {idx}', user_id='u1', agent_id='ag')
        with _open(db_path) as conn:
            oldest = conn.execute(
                'SELECT user_id, agent_id, hash FROM memories ORDER BY id ASC LIMIT 1').fetchone()
        m._active_writes.add((oldest['user_id'], oldest['agent_id'], oldest['hash']))
        report = m.cleanup()
        assert report['deleted'] == 1
        assert [row['query'] for row in _rows(db_path)] == [
            'active question 0', 'active question 2', 'active question 3']


class TestLocalMemoryResilience:
    def test_corrupt_database_file_is_self_healed(self, db_path):
        with open(db_path, 'wb') as f:
            f.write(b'not a real sqlite database' * 200)
        m = LocalFileMemory(db_path)
        m.add('recovery question', 'recovery answer', user_id='u1', agent_id='a1')
        assert 'recovery answer' in m.get('recovery', user_id='u1', agent_id='a1')
        quarantined = [name for name in os.listdir(os.path.dirname(db_path)) if '.corrupt-' in name]
        assert quarantined

    def test_corrupt_row_is_skipped_and_purged(self, db_path):
        m = LocalFileMemory(db_path, auto_cleanup=False)
        m.add('good question', 'good answer', user_id='u1', agent_id='a1')
        m.add('bad question', 'bad answer', user_id='u1', agent_id='a1')
        with _open(db_path) as conn:
            conn.execute("UPDATE memories SET message = '{broken-json' WHERE query = 'bad question'")
        result = m.get('good bad', user_id='u1', agent_id='a1')
        assert 'good answer' in result
        assert 'bad answer' not in result
        report = m.cleanup()
        assert report['deleted'] == 1
        assert report['reasons']['corrupt'] == 1

    def test_unwritable_path_does_not_raise(self, tmp_path):
        blocker = tmp_path / 'regular_file'
        blocker.write_text('x')
        m = LocalMemory(str(blocker / 'subdir' / 'x.db'))
        m.add('failing question', 'failing answer', user_id='u1', agent_id='a1')
        assert m.get('failing', user_id='u1', agent_id='a1') == ''

    def test_concurrent_writes(self, db_path):
        m = LocalFileMemory(db_path, auto_cleanup=False)
        n_threads, n_per_thread = 8, 30

        def worker(tid):
            for idx in range(n_per_thread):
                m.add(f'concurrent question {tid}-{idx}', f'answer {tid}-{idx}',
                      user_id=f'u{tid % 3}', agent_id='ag')

        threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(n_threads)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        rows = _rows(db_path)
        assert len(rows) == n_threads * n_per_thread
        assert len({row['hash'] for row in rows}) == n_threads * n_per_thread
        assert m.get('concurrent', user_id='u0', agent_id='ag') != ''


class TestLocalMemoryEmbedding:
    def test_embedding_adapter_reranks(self, db_path):
        vocab = {'cat': [1.0, 0.0, 0.0], 'dog': [0.9, 0.1, 0.0], 'rocket': [0.0, 0.0, 1.0]}

        def embed(text):
            for key, vec in vocab.items():
                if key in text:
                    return vec
            return [0.0, 1.0, 0.0]

        m = LocalFileMemory(db_path, embed=embed, topk=1)
        m.add('tell me about the cat', 'cats purr', user_id='u1', agent_id='a1')
        m.add('how do rockets launch', 'with fuel', user_id='u1', agent_id='a1')
        m.add('random weather chatter', 'sunny', user_id='u1', agent_id='a1')
        result = m.get('feline kitten cat', user_id='u1', agent_id='a1')
        assert 'cats purr' in result
        with _open(db_path) as conn:
            stored = conn.execute('SELECT embedding FROM memories LIMIT 1').fetchone()['embedding']
        assert json.loads(stored) == [1.0, 0.0, 0.0]

    def test_broken_embedding_falls_back_to_keyword(self, db_path):
        def broken_embed(text):
            raise RuntimeError('embedding service offline')

        m = LocalFileMemory(db_path, embed=broken_embed)
        m.add('offline keyword question', 'still retrievable', user_id='u1', agent_id='a1')
        assert 'still retrievable' in m.get('keyword', user_id='u1', agent_id='a1')
        assert _rows(db_path)[0]['embedding'] is None


class TestMemoryFactorySelection:
    def test_defaults_to_local_without_external_keys(self):
        assert isinstance(Memory(), LocalFileMemory)

    def test_external_key_takes_priority(self, monkeypatch):
        class DummySupplier:
            def __init__(self, *args, **kwargs):
                self.args = args

        monkeypatch.setitem(Memory.SUPPLIERS, 'memu', DummySupplier)
        monkeypatch.setitem(Memory.SUPPLIERS, 'mem0', DummySupplier)
        with lazyllm.config.temp('memu_api_key', 'fake-memu-key'):
            assert isinstance(Memory(), DummySupplier)
        with lazyllm.config.temp('mem0_api_key', 'fake-mem0-key'):
            assert isinstance(Memory(), DummySupplier)

    def test_powermem_remains_explicitly_selectable(self):
        assert 'powermem' in Memory.SUPPLIERS
        assert 'local' in Memory.SUPPLIERS


class TestMemoryHookIntegration:
    def test_hook_persists_turn_through_local_supplier(self, db_path, monkeypatch):
        monkeypatch.setattr(Memory, 'SUPPLIERS', dict(Memory.SUPPLIERS))
        monkeypatch.setitem(Memory.SUPPLIERS, 'local',
                            lambda *args, **kwargs: LocalFileMemory(db_path))
        old_user, old_agent = lazyllm.globals.get('user_id'), lazyllm.globals.get('agent_id')
        try:
            with lazyllm.config.temp('local_memory_path', db_path):
                lazyllm.globals['user_id'] = 'hook-user'
                lazyllm.globals['agent_id'] = 'hook-agent'
                gen = memory_hook('hook question about sailing')
                next(gen)
                with pytest.raises(StopIteration):
                    gen.send('sailing needs wind')
                m = LocalFileMemory(db_path)
                result = m.get('sailing', user_id='hook-user', agent_id='hook-agent')
                assert 'sailing needs wind' in result
                assert 'hook question about sailing' in result
                assert m() == result
        finally:
            lazyllm.globals['user_id'] = old_user
            lazyllm.globals['agent_id'] = old_agent
