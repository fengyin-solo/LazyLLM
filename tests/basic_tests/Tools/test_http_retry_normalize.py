import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest

from lazyllm.thirdparty import httpx
from lazyllm.tools.http_request.http_request import (
    HttpRequest,
    RetryConfig,
    HttpRequestError,
    HttpResponseTooLargeError,
)
import lazyllm.tools.http_request.http_request as http_request_module

URL = 'https://example.com/api'
FILE_URL = 'https://example.com/files/photo.png'


class FakeStreamResponse:
    def __init__(self, outcome, request):
        self.status_code = outcome['status']
        self.headers = outcome.get('headers') or {}
        self._chunks = outcome.get('chunks') or []
        self._read_error = outcome.get('read_error')
        self.request = request
        self.body_started = False

    def iter_bytes(self):
        for index, chunk in enumerate(self._chunks):
            self.body_started = True
            if self._read_error is not None and index >= 1:
                raise self._read_error
            yield chunk


def install_fake_stream(monkeypatch, handler):
    calls = []
    responses = []
    lock = threading.Lock()

    @contextmanager
    def fake_stream(**kwargs):
        with lock:
            calls.append(kwargs)
            outcome = handler(kwargs)
        if isinstance(outcome, BaseException):
            raise outcome
        response = FakeStreamResponse(outcome, httpx.Request(kwargs['method'], kwargs['url']))
        responses.append(response)
        yield response

    monkeypatch.setattr(http_request_module.httpx, 'stream', fake_stream)
    return calls, responses


def outcome(status, body=b'', content_type=None, headers=None, read_error=None):
    merged_headers = dict(headers or {})
    if content_type:
        merged_headers['content-type'] = content_type
    chunks = [body] if isinstance(body, (bytes, bytearray)) else list(body)
    return {'status': status, 'headers': merged_headers, 'chunks': chunks, 'read_error': read_error}


def make_request(method='GET', url=URL, retry=None, max_download_bytes=None, timeout=10):
    kwargs = {}
    if max_download_bytes is not None:
        kwargs['max_download_bytes'] = max_download_bytes
    return HttpRequest(method, url, api_key='', headers={}, params={}, body='',
                       timeout=timeout, retry=retry, **kwargs)


class TestLegacyBehavior:
    def test_legacy_shape_and_single_request(self, monkeypatch):
        calls = []

        def fake_request(**kwargs):
            calls.append(kwargs)
            return httpx.Response(500, headers={'content-type': 'text/plain'}, content=b'busy')

        monkeypatch.setattr(http_request_module.httpx, 'request', fake_request)

        result = HttpRequest('GET', URL, api_key='', headers={}, params={}, body='')()

        assert len(calls) == 1
        assert set(result.keys()) == {'status_code', 'content', 'headers', 'file'}
        assert result['status_code'] == 500
        assert result['content'] == 'busy'
        assert result['file'] == b''

    def test_legacy_does_not_retry_or_wrap_timeout(self, monkeypatch):
        calls = []

        def fake_request(**kwargs):
            calls.append(kwargs)
            raise httpx.ReadTimeout('read timed out')

        monkeypatch.setattr(http_request_module.httpx, 'request', fake_request)

        request = HttpRequest('GET', URL, api_key='', headers={}, params={}, body='')
        with pytest.raises(httpx.TimeoutException):
            request.forward()
        assert len(calls) == 1


class TestRetry:
    def test_retry_on_5xx_then_success(self, monkeypatch):
        responses = [outcome(503, b'busy', 'text/plain'),
                     outcome(503, b'busy', 'text/plain'),
                     outcome(200, b'{"ok": true}', 'application/json')]
        calls, _ = install_fake_stream(monkeypatch, lambda kwargs: responses[len(calls) - 1])

        result = make_request(retry={'max_attempts': 3, 'base_delay': 0, 'jitter': 0})()

        assert len(calls) == 3
        assert set(result.keys()) == {'status_code', 'content_type', 'headers', 'text', 'json', 'file'}
        assert result['status_code'] == 200
        assert result['json'] == {'ok': True}
        assert result['file'] is None

    def test_retry_exhausted_raises_unified_error(self, monkeypatch):
        calls, _ = install_fake_stream(
            monkeypatch, lambda kwargs: outcome(500, b'error', 'text/plain'))

        with pytest.raises(HttpRequestError) as exc_info:
            make_request(retry={'max_attempts': 3, 'base_delay': 0, 'jitter': 0}).forward()

        error = exc_info.value
        assert error.attempt_count == 3
        assert error.last_status == 500
        assert error.method == 'GET'
        assert error.url == URL
        assert [record['attempt'] for record in error.attempts] == [1, 2, 3]
        assert [record['status'] for record in error.attempts] == [500, 500, 500]
        assert all(isinstance(record['elapsed'], float) for record in error.attempts)
        assert len(calls) == 3

    def test_timeout_is_retried_and_normalized(self, monkeypatch):
        calls, _ = install_fake_stream(
            monkeypatch, lambda kwargs: httpx.ReadTimeout('read timed out'))

        with pytest.raises(HttpRequestError) as exc_info:
            make_request(retry={'max_attempts': 3, 'base_delay': 0, 'jitter': 0}).forward()

        error = exc_info.value
        assert error.attempt_count == 3
        assert error.last_status is None
        assert all('timed out' in record['error'] for record in error.attempts)
        assert len(calls) == 3

    def test_network_error_retried_then_success(self, monkeypatch):
        responses = [httpx.ConnectError('connection reset'),
                     httpx.ConnectError('connection reset'),
                     outcome(200, b'hello', 'text/plain')]
        calls, _ = install_fake_stream(monkeypatch, lambda kwargs: responses[len(calls) - 1])

        result = make_request(retry={'max_attempts': 3, 'base_delay': 0, 'jitter': 0})()

        assert len(calls) == 3
        assert result['status_code'] == 200
        assert result['text'] == 'hello'

    def test_read_timeout_mid_body_is_retried(self, monkeypatch):
        responses = [outcome(200, [b'a', b'b'], 'application/octet-stream',
                             read_error=httpx.ReadTimeout('mid-body timeout')),
                     outcome(200, b'done', 'text/plain')]
        calls, _ = install_fake_stream(monkeypatch, lambda kwargs: responses[len(calls) - 1])

        result = make_request(retry={'max_attempts': 2, 'base_delay': 0, 'jitter': 0})()

        assert len(calls) == 2
        assert result['text'] == 'done'

    def test_4xx_returned_without_retry(self, monkeypatch):
        calls, _ = install_fake_stream(
            monkeypatch, lambda kwargs: outcome(404, b'not found', 'text/plain'))

        result = make_request(retry=True)()

        assert result['status_code'] == 404
        assert result['text'] == 'not found'
        assert len(calls) == 1

    def test_non_idempotent_post_not_retried_on_5xx(self, monkeypatch):
        calls, _ = install_fake_stream(
            monkeypatch, lambda kwargs: outcome(503, b'busy', 'text/plain'))

        result = make_request(method='POST', retry={'max_attempts': 3, 'base_delay': 0})()

        assert result['status_code'] == 503
        assert len(calls) == 1

    def test_non_idempotent_post_not_retried_on_timeout(self, monkeypatch):
        calls, _ = install_fake_stream(
            monkeypatch, lambda kwargs: httpx.ReadTimeout('read timed out'))

        with pytest.raises(HttpRequestError) as exc_info:
            make_request(method='POST', retry={'max_attempts': 3, 'base_delay': 0}).forward()

        assert exc_info.value.attempt_count == 1
        assert len(calls) == 1

    def test_opt_in_retry_for_post(self, monkeypatch):
        responses = [outcome(503, b'busy', 'text/plain'),
                     outcome(200, b'created', 'text/plain')]
        calls, _ = install_fake_stream(monkeypatch, lambda kwargs: responses[len(calls) - 1])

        config = RetryConfig(max_attempts=2, base_delay=0, jitter=0,
                             idempotent_methods=('GET', 'POST'))
        result = make_request(method='POST', retry={'max_attempts': 2, 'base_delay': 0,
                                                    'jitter': 0,
                                                    'idempotent_methods': ('GET', 'POST')})()
        assert config.is_idempotent('post')
        assert result['status_code'] == 200
        assert len(calls) == 2

    def test_custom_retryable_statuses(self, monkeypatch):
        calls, _ = install_fake_stream(
            monkeypatch, lambda kwargs: outcome(521 if len(calls) == 1 else 200,
                                                b'x', 'text/plain'))

        result = make_request(retry={'max_attempts': 2, 'base_delay': 0, 'jitter': 0,
                                     'retryable_statuses': (521,)})()

        assert result['status_code'] == 200
        assert len(calls) == 2

    def test_unlisted_5xx_not_retried(self, monkeypatch):
        calls, _ = install_fake_stream(
            monkeypatch, lambda kwargs: outcome(500, b'err', 'text/plain'))

        result = make_request(retry={'max_attempts': 3, 'base_delay': 0, 'jitter': 0,
                                     'retryable_statuses': (521,)})()

        assert result['status_code'] == 500
        assert len(calls) == 1

    def test_exponential_backoff_without_jitter(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr(http_request_module.time, 'sleep', lambda seconds: sleeps.append(seconds))
        responses = [outcome(503, b'busy', 'text/plain'),
                     outcome(503, b'busy', 'text/plain'),
                     outcome(200, b'ok', 'text/plain')]
        calls, _ = install_fake_stream(monkeypatch, lambda kwargs: responses[len(calls) - 1])

        make_request(retry={'max_attempts': 3, 'base_delay': 0.5, 'max_delay': 30, 'jitter': 0})()

        assert len(calls) == 3
        assert sleeps == pytest.approx([0.5, 1.0])

    def test_backoff_respects_max_delay(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr(http_request_module.time, 'sleep', lambda seconds: sleeps.append(seconds))
        responses = [outcome(503, b'busy', 'text/plain')] * 3 + [outcome(200, b'ok', 'text/plain')]
        calls, _ = install_fake_stream(monkeypatch, lambda kwargs: responses[len(calls) - 1])

        make_request(retry={'max_attempts': 4, 'base_delay': 1.0, 'max_delay': 1.5,
                            'jitter': 0})()

        assert sleeps == pytest.approx([1.0, 1.5, 1.5])

    def test_jitter_applied(self, monkeypatch):
        monkeypatch.setattr(http_request_module.random, 'uniform',
                            lambda low, high: high)
        sleeps = []
        monkeypatch.setattr(http_request_module.time, 'sleep', lambda seconds: sleeps.append(seconds))
        responses = [outcome(503, b'busy', 'text/plain'),
                     outcome(503, b'busy', 'text/plain'),
                     outcome(200, b'ok', 'text/plain')]
        calls, _ = install_fake_stream(monkeypatch, lambda kwargs: responses[len(calls) - 1])

        make_request(retry={'max_attempts': 3, 'base_delay': 1.0, 'max_delay': 30,
                            'jitter': 0.5})()

        assert sleeps == pytest.approx([1.5, 3.0])


class TestNormalizedResponse:
    def test_json_response(self, monkeypatch):
        calls, _ = install_fake_stream(
            monkeypatch, lambda kwargs: outcome(200, b'{"a": 1, "b": [2, 3]}',
                                                'application/json; charset=utf-8'))

        result = make_request(retry={'max_attempts': 1})()

        assert result['status_code'] == 200
        assert result['content_type'] == 'application/json'
        assert result['json'] == {'a': 1, 'b': [2, 3]}
        assert result['text'] == '{"a": 1, "b": [2, 3]}'
        assert result['file'] is None
        assert result['headers']['content-type'] == 'application/json; charset=utf-8'

    def test_text_response(self, monkeypatch):
        calls, _ = install_fake_stream(
            monkeypatch, lambda kwargs: outcome(200, b'<h1>hi</h1>', 'text/html'))

        result = make_request(retry={'max_attempts': 1})()

        assert result['content_type'] == 'text/html'
        assert result['text'] == '<h1>hi</h1>'
        assert result['json'] is None

    def test_malformed_json_body(self, monkeypatch):
        calls, _ = install_fake_stream(
            monkeypatch, lambda kwargs: outcome(200, b'not-json', 'application/json'))

        result = make_request(retry={'max_attempts': 1})()

        assert result['json'] is None
        assert result['text'] == 'not-json'

    def test_file_by_content_type_and_url_suffix(self, monkeypatch):
        png_bytes = b'\x89PNG\r\n\x1a\n' + b'0' * 20
        calls, _ = install_fake_stream(
            monkeypatch, lambda kwargs: outcome(200, png_bytes, 'image/png'))

        result = make_request(url=FILE_URL, retry={'max_attempts': 1})()

        assert result['file'] == {'filename': 'photo.png', 'content_type': 'image/png',
                                  'content': png_bytes}
        assert result['text'] is None
        assert result['json'] is None

    def test_file_by_content_disposition_filename(self, monkeypatch):
        body = b'binary-data'
        headers = {'content-disposition': "attachment; filename=report.bin"}
        calls, _ = install_fake_stream(
            monkeypatch, lambda kwargs: outcome(200, body, 'application/octet-stream', headers))

        result = make_request(url='https://example.com/download', retry={'max_attempts': 1})()

        assert result['file']['filename'] == 'report.bin'
        assert result['file']['content'] == body

    def test_file_by_rfc5987_disposition_filename(self, monkeypatch):
        body = b'binary-data'
        headers = {'content-disposition': "attachment; filename*=UTF-8''%E6%8A%A5%E5%91%8A.bin"}
        calls, _ = install_fake_stream(
            monkeypatch, lambda kwargs: outcome(200, body, 'application/octet-stream', headers))

        result = make_request(url='https://example.com/download', retry={'max_attempts': 1})()

        assert result['file']['filename'] == '报告.bin'

    def test_default_filename_from_content_type(self, monkeypatch):
        body = b'binary-data'
        calls, _ = install_fake_stream(
            monkeypatch, lambda kwargs: outcome(200, body, 'image/gif'))

        result = make_request(url='https://example.com/random-endpoint',
                              retry={'max_attempts': 1})()

        assert result['file']['filename'] == 'download.gif'
        assert result['file']['content'] == body


class TestDownloadLimit:
    def test_reject_when_content_length_too_large(self, monkeypatch):
        calls, responses = install_fake_stream(
            monkeypatch,
            lambda kwargs: outcome(200, b'x' * 1000, 'application/octet-stream',
                                   {'content-length': '1000'}))

        with pytest.raises(HttpResponseTooLargeError) as exc_info:
            make_request(retry={'max_attempts': 3, 'base_delay': 0},
                         max_download_bytes=512).forward()

        error = exc_info.value
        assert error.max_bytes == 512
        assert error.last_status == 200
        assert error.attempt_count == 1
        assert len(calls) == 1
        assert responses[0].body_started is False

    def test_reject_when_streamed_body_too_large(self, monkeypatch):
        chunks = [b'x' * 200 for _ in range(5)]
        calls, _ = install_fake_stream(
            monkeypatch,
            lambda kwargs: outcome(200, chunks, 'application/octet-stream'))

        with pytest.raises(HttpResponseTooLargeError) as exc_info:
            make_request(retry={'max_attempts': 3, 'base_delay': 0},
                         max_download_bytes=450).forward()

        error = exc_info.value
        assert error.downloaded > 450
        assert len(calls) == 1

    def test_body_within_limit_accepted(self, monkeypatch):
        calls, _ = install_fake_stream(
            monkeypatch, lambda kwargs: outcome(200, b'x' * 100, 'application/octet-stream',
                                                {'content-length': '100'}))

        result = make_request(retry={'max_attempts': 1}, max_download_bytes=500)()

        assert result['status_code'] == 200
        assert len(result['file']['content']) == 100


class TestConcurrency:
    def test_concurrent_requests_isolated_attempts(self, monkeypatch):
        task_state = threading.local()
        lock = threading.Lock()
        total_calls = {'value': 0}

        def handler(kwargs):
            first_attempt = not getattr(task_state, 'retry_seen', False)
            task_state.retry_seen = True
            with lock:
                total_calls['value'] += 1
            return outcome(503 if first_attempt else 200, b'{"ok": true}', 'application/json')

        calls, _ = install_fake_stream(monkeypatch, handler)
        request = make_request(retry={'max_attempts': 2, 'base_delay': 0, 'jitter': 0})

        def run_task(index):
            task_state.retry_seen = False
            return request.forward()

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(run_task, range(8)))

        assert len(results) == 8
        assert all(result['status_code'] == 200 for result in results)
        assert all(result['json'] == {'ok': True} for result in results)
        assert len(calls) == 16
        assert total_calls['value'] == 16

    def test_module_call_wraps_unified_error(self, monkeypatch):
        from lazyllm.module.module import ModuleExecutionError

        calls, _ = install_fake_stream(
            monkeypatch, lambda kwargs: outcome(503, b'busy', 'text/plain'))

        request = make_request(retry={'max_attempts': 2, 'base_delay': 0, 'jitter': 0})
        with pytest.raises(ModuleExecutionError) as exc_info:
            request()

        message = str(exc_info.value)
        assert '503' in message
        assert '2 attempt(s)' in message
        assert len(calls) == 2
