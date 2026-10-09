import json
import random
import re
import time

import requests
from lazyllm.thirdparty import httpx
from lazyllm.module.module import ModuleBase
from lazyllm.tools.http_request.http_executor_response import HttpExecutorResponse
from lazyllm import LOG

DEFAULT_MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024
DEFAULT_IDEMPOTENT_METHODS = ('GET', 'HEAD', 'PUT', 'DELETE', 'OPTIONS')
DEFAULT_RETRYABLE_STATUSES = (408, 429, 500, 502, 503, 504)


class HttpRequestError(Exception):
    def __init__(self, message, *, method=None, url=None, attempts=None, last_status=None):
        super().__init__(message)
        self.method = method
        self.url = url
        self.attempts = list(attempts or [])
        self.last_status = last_status
        self.attempt_count = len(self.attempts)


class HttpResponseTooLargeError(HttpRequestError):
    def __init__(self, message, *, max_bytes=None, downloaded=None, **kwargs):
        super().__init__(message, **kwargs)
        self.max_bytes = max_bytes
        self.downloaded = downloaded


class _ResponseTooLarge(Exception):
    def __init__(self, max_bytes, downloaded, status):
        self.max_bytes = max_bytes
        self.downloaded = downloaded
        self.status = status


class RetryConfig:
    def __init__(self, max_attempts=3, base_delay=0.5, max_delay=30.0, jitter=0.1,
                 retryable_statuses=DEFAULT_RETRYABLE_STATUSES,
                 idempotent_methods=DEFAULT_IDEMPOTENT_METHODS):
        max_attempts = int(max_attempts)
        if max_attempts < 1:
            raise ValueError('max_attempts must be greater than or equal to 1')
        if base_delay < 0 or max_delay < 0:
            raise ValueError('delay values must be non-negative')
        if not 0 <= jitter <= 1:
            raise ValueError('jitter must be within [0, 1]')

        self.max_attempts = max_attempts
        self.base_delay = float(base_delay)
        self.max_delay = float(max_delay)
        self.jitter = float(jitter)
        self.retryable_statuses = frozenset(retryable_statuses)
        self.idempotent_methods = frozenset(m.upper() for m in idempotent_methods)

    def is_idempotent(self, method) -> bool:
        return (method or '').upper() in self.idempotent_methods

    def is_retryable_status(self, status) -> bool:
        return status in self.retryable_statuses

    def backoff_delay(self, attempt: int) -> float:
        delay = min(self.max_delay, self.base_delay * (2 ** max(0, attempt - 1)))
        if self.jitter:
            delay = max(0.0, delay * (1.0 + random.uniform(-self.jitter, self.jitter)))
        return delay


class HttpRequest(ModuleBase):
    def __init__(self, method, url, api_key, headers, params, body, timeout=10, proxies=None,
                 retry=None, max_download_bytes=DEFAULT_MAX_DOWNLOAD_BYTES):
        super().__init__()
        if not url:
            return

        self._method = method
        self._url = url
        self._api_key = api_key
        self._headers = headers
        self._params = params
        self._body = body
        self._timeout = timeout
        self._proxies = proxies
        self._retry_config = self._build_retry_config(retry)
        self._max_download_bytes = max_download_bytes

    @staticmethod
    def _build_retry_config(retry):
        if not retry:
            return None
        if isinstance(retry, RetryConfig):
            return retry
        if retry is True:
            return RetryConfig()
        if isinstance(retry, dict):
            return RetryConfig(**retry)
        raise TypeError('retry must be a RetryConfig, a dict, a boolean or None')

    def _process_api_key(self, headers, params):
        if self._api_key and self._api_key != '':
            params = params or {}
            params['api_key'] = self._api_key
        return headers, params

    def forward(self, *args, **kwargs):
        def _map_input(target_str):
            if not isinstance(target_str, str):
                return target_str

            # TODO: replacements could be more complex to create.
            replacements = {**kwargs, **(args[0] if args and isinstance(args[0], dict) else {})}
            if not replacements:
                return target_str

            pattern = r'\{\{([^}]+)\}\}'

            full_match = re.fullmatch(pattern, target_str)
            if full_match:
                key = full_match.group(1)
                if key in replacements:
                    return replacements[key]

            def replacer(m):
                key = m.group(1)
                if key not in replacements:
                    return m.group(0)  # Keep original if no replacement found
                replacement = replacements[key]
                if isinstance(replacement, (dict, list, bool)) or replacement is None:
                    return json.dumps(replacement, ensure_ascii=False)
                return str(replacement)

            return re.sub(pattern, replacer, target_str)

        url = _map_input(self._url)
        params = {key: _map_input(value) for key, value in self._params.items()} if self._params else None
        headers = {key: _map_input(value) for key, value in self._headers.items()} if self._headers else None
        headers, params = self._process_api_key(headers, params)

        request_kwargs = self._build_request_kwargs(url, headers, params, _map_input)
        if self._retry_config is None:
            return self._send_once_legacy(request_kwargs)
        return self._send_with_retry(request_kwargs)

    def _build_request_kwargs(self, url, headers, params, map_input):
        request_kwargs = dict(method=self._method, url=url, headers=headers, params=params,
                              timeout=self._timeout, proxies=self._proxies)
        if isinstance(headers, dict) and headers.get('Content-Type') == 'application/json':
            try:
                body = json.dumps(self._body) if isinstance(self._body, dict) else self._body
                body = json.loads(map_input(body))
            except json.JSONDecodeError:
                raise ValueError(f'Invalid JSON format: {self._body}')
            request_kwargs['json'] = body
        else:
            body = (json.dumps({k: map_input(v) for k, v in self._body.items()})
                    if isinstance(self._body, dict) else map_input(self._body))
            request_kwargs['data'] = body
        return request_kwargs

    def _send_once_legacy(self, request_kwargs):
        http_response = httpx.request(**request_kwargs)

        response = HttpExecutorResponse(http_response)

        _, file_binary = response.extract_file()

        outputs = {
            'status_code': response.status_code,
            'content': response.content if len(file_binary) == 0 else None,
            'headers': response.headers,
            'file': file_binary
        }
        return outputs

    def _read_response(self, request_kwargs):
        max_bytes = self._max_download_bytes
        with httpx.stream(**request_kwargs) as resp:
            if max_bytes is not None:
                content_length = resp.headers.get('content-length')
                if content_length and content_length.isdigit() and int(content_length) > max_bytes:
                    raise _ResponseTooLarge(max_bytes, 0, resp.status_code)

            body = bytearray()
            for chunk in resp.iter_bytes():
                body.extend(chunk)
                if max_bytes is not None and len(body) > max_bytes:
                    raise _ResponseTooLarge(max_bytes, len(body), resp.status_code)

            return httpx.Response(resp.status_code, headers=dict(resp.headers),
                                  content=bytes(body), request=resp.request)

    def _send_with_retry(self, request_kwargs):
        config = self._retry_config
        method = (self._method or '').upper()
        url = request_kwargs['url']
        attempts = []

        while True:
            attempt_no = len(attempts) + 1
            started = time.monotonic()
            response = None
            error = None
            try:
                response = self._read_response(request_kwargs)
            except _ResponseTooLarge as e:
                elapsed = time.monotonic() - started
                attempts.append({'attempt': attempt_no, 'elapsed': elapsed,
                                 'status': e.status, 'error': 'response too large'})
                message = (f'[HttpRequest] {method} {url} response exceeds max_download_bytes='
                           f'{e.max_bytes} (downloaded={e.downloaded})')
                LOG.error(message)
                raise HttpResponseTooLargeError(
                    message, method=method, url=url, attempts=attempts, last_status=e.status,
                    max_bytes=e.max_bytes, downloaded=e.downloaded) from None
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                error = e
            except httpx.RequestError as e:
                error = e

            elapsed = time.monotonic() - started
            status = response.status_code if response is not None else None
            transient = isinstance(error, (httpx.TimeoutException, httpx.NetworkError))
            retryable_status = response is not None and config.is_retryable_status(status)
            can_retry = (attempt_no < config.max_attempts and config.is_idempotent(method)
                         and (transient or retryable_status))

            attempts.append({'attempt': attempt_no, 'elapsed': elapsed,
                             'status': status, 'error': str(error) if error else None})
            result_desc = f'status={status}' if error is None else f'error={error}'
            LOG.warning(f'[HttpRequest] {method} {url} attempt {attempt_no}/{config.max_attempts} '
                        f'finished in {elapsed:.3f}s with {result_desc}')

            if can_retry:
                delay = config.backoff_delay(attempt_no)
                LOG.warning(f'[HttpRequest] {method} {url} transient failure, retrying '
                            f'attempt {attempt_no + 1}/{config.max_attempts} after {delay:.3f}s')
                time.sleep(delay)
                continue

            if error is not None:
                message = (f'[HttpRequest] {method} {url} failed after {attempt_no} attempt(s) '
                           f'with last_status={status}: {error}')
                LOG.error(message)
                raise HttpRequestError(message, method=method, url=url,
                                       attempts=attempts, last_status=status) from error

            if retryable_status and config.is_idempotent(method):
                message = (f'[HttpRequest] {method} {url} returned status {status} after '
                           f'{attempt_no} attempt(s)')
                LOG.error(message)
                raise HttpRequestError(message, method=method, url=url,
                                       attempts=attempts, last_status=status)

            return self._normalize(response, url)

    @staticmethod
    def _normalize(response, url):
        executor = HttpExecutorResponse(response)
        file_info = executor.extract_file_info(url)

        return {
            'status_code': executor.status_code,
            'content_type': executor.content_type,
            'headers': executor.headers,
            'text': None if file_info else executor.text,
            'json': executor.json(),
            'file': file_info,
        }


def post_sync(url: str, payload: dict = None, files: dict = None, headers: dict = None,
              json_payload: dict = None, timeout: int = 60,
              raise_for_status: bool = True) -> requests.Response:
    '''Execute a synchronous POST request with unified error handling.'''
    try:
        if json_payload is not None:
            resp = requests.post(url, json=json_payload, headers=headers, timeout=timeout)
        elif files is not None:
            resp = requests.post(url, data=payload, files=files, headers=headers, timeout=timeout)
        else:
            resp = requests.post(url, data=payload, headers=headers, timeout=timeout)
        if raise_for_status:
            resp.raise_for_status()
        return resp
    except requests.exceptions.RequestException as e:
        LOG.error(f'[HttpRequest] POST request to {url} failed: {e}')
        raise


def get_sync(url: str, headers: dict = None, timeout: int = 60,
             raise_for_status: bool = True) -> requests.Response:
    '''Execute a synchronous GET request with unified error handling.'''
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        if raise_for_status:
            resp.raise_for_status()
        return resp
    except requests.exceptions.RequestException as e:
        LOG.error(f'[HttpRequest] GET request to {url} failed: {e}')
        raise
