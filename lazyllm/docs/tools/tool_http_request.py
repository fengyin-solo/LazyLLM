# flake8: noqa E501
import importlib
import functools
from .. import utils
add_chinese_doc = functools.partial(utils.add_chinese_doc, module=importlib.import_module('lazyllm.tools'))
add_english_doc = functools.partial(utils.add_english_doc, module=importlib.import_module('lazyllm.tools'))
add_example = functools.partial(utils.add_example, module=importlib.import_module('lazyllm.tools'))

add_chinese_doc('http_request.http_executor_response.HttpExecutorResponse', """\
HTTP执行器响应类，用于封装和处理HTTP请求的响应结果。

提供对HTTP响应内容的统一访问接口，支持文件类型检测和内容提取。

Args:
    response (httpx.Response, optional): httpx库的响应对象，默认为None


**Returns:**\n
- HttpExecutorResponse实例，提供多种响应内容访问方式
""")

add_english_doc('http_request.http_executor_response.HttpExecutorResponse', """\
HTTP executor response class for encapsulating and processing HTTP request response results.

Provides unified access interface for HTTP response content, supporting file type detection and content extraction.

Args:
    response (httpx.Response, optional): httpx library response object, defaults to None

**Returns:**\n
- HttpExecutorResponse instance, providing multiple response content access methods
""")

add_chinese_doc('http_request.http_executor_response.HttpExecutorResponse.get_content_type', '''\
获取HTTP响应的内容类型。

从响应头中提取 'content-type' 字段的值，用于判断响应内容的类型。

**Returns:**\n
- str: 响应的内容类型，如果未找到则返回空字符串。
''')

add_english_doc('http_request.http_executor_response.HttpExecutorResponse.get_content_type', '''\
Get the content type of the HTTP response.

Extracts the 'content-type' field value from the response headers to determine the type of response content.

**Returns:**\n
- str: The content type of the response, or empty string if not found.
''')

add_example('http_request.http_executor_response.HttpExecutorResponse.get_content_type', '''\
>>> from lazyllm.tools.http_request.http_executor_response import HttpExecutorResponse
>>> import httpx
>>> response = httpx.Response(200, headers={'content-type': 'application/json'})
>>> http_response = HttpExecutorResponse(response)
>>> content_type = http_response.get_content_type()
>>> print(content_type)
... 'application/json'
''')

add_chinese_doc('http_request.http_executor_response.HttpExecutorResponse.extract_file', '''\
从HTTP响应中提取文件内容。

如果响应内容类型是文件相关类型（如图片、音频、视频），则提取文件的内容类型和二进制数据。

**Returns:**\n
- tuple[str, bytes]: 包含内容类型和文件二进制数据的元组。如果不是文件类型，则返回空字符串和空字节。
''')

add_english_doc('http_request.http_executor_response.HttpExecutorResponse.extract_file', '''\
Extract file content from HTTP response.

If the response content type is file-related (such as image, audio, video), extracts the content type and binary data of the file.

**Returns:**\n
- tuple[str, bytes]: A tuple containing the content type and binary data of the file. If not a file type, returns empty string and empty bytes.
''')

add_example('http_request.http_executor_response.HttpExecutorResponse.extract_file', '''\
>>> from lazyllm.tools.http_request.http_executor_response import HttpExecutorResponse
>>> import httpx
>>> # 模拟图片响应
>>> response = httpx.Response(200, headers={'content-type': 'image/jpeg'}, content=b'fake_image_data')
>>> http_response = HttpExecutorResponse(response)
>>> content_type, file_data = http_response.extract_file()
>>> print(content_type)
... 'image/jpeg'
>>> print(len(file_data))
... 15
>>> # 模拟JSON响应
>>> response = httpx.Response(200, headers={'content-type': 'application/json'}, content=b'{"key": "value"}')
>>> http_response = HttpExecutorResponse(response)
>>> content_type, file_data = http_response.extract_file()
>>> print(content_type)
... ''
>>> print(file_data)
... b''
''')

add_chinese_doc('http_request.http_request.RetryConfig', '''\
HttpRequest 的自动重试配置。

仅对幂等方法（默认 GET/HEAD/PUT/DELETE/OPTIONS）的超时、网络错误以及可重试状态码（默认
408/429/500/502/503/504）进行重试，退避延迟按指数增长并可叠加抖动。

Args:
    max_attempts (int): 包含首次请求在内的最大尝试次数，默认 3。
    base_delay (float): 首次重试前的基础退避秒数，默认 0.5。
    max_delay (float): 退避秒数上限，默认 30。
    jitter (float): 抖动比例，取值 [0, 1]，默认 0.1；设为 0 表示退避完全确定。
    retryable_statuses (tuple[int]): 触发重试的 HTTP 状态码集合。
    idempotent_methods (tuple[str]): 视为幂等、允许重试的 HTTP 方法集合。
''')

add_english_doc('http_request.http_request.RetryConfig', '''\
Automatic retry configuration for HttpRequest.

Only idempotent methods (GET/HEAD/PUT/DELETE/OPTIONS by default) are retried, and only on
timeouts, network errors or retryable status codes (408/429/500/502/503/504 by default).
Backoff grows exponentially and can be jittered.

Args:
    max_attempts (int): Maximum number of attempts including the first request, defaults to 3.
    base_delay (float): Base backoff in seconds before the first retry, defaults to 0.5.
    max_delay (float): Upper bound of the backoff in seconds, defaults to 30.
    jitter (float): Jitter ratio within [0, 1], defaults to 0.1; 0 makes backoff deterministic.
    retryable_statuses (tuple[int]): HTTP status codes that trigger a retry.
    idempotent_methods (tuple[str]): HTTP methods treated as idempotent and retryable.
''')

add_chinese_doc('http_request.http_request.HttpRequestError', '''\
归一化后的 HTTP 请求失败异常。

重试耗尽或发生不可重试的请求异常时抛出，携带 method、url、last_status 以及每次尝试的
attempts 记录（包含 attempt、elapsed、status、error 字段）。
''')

add_english_doc('http_request.http_request.HttpRequestError', '''\
Normalized HTTP request failure exception.

Raised when retries are exhausted or a non-retryable request error occurs. Carries method,
url, last_status and per-attempt records (attempt, elapsed, status, error).
''')

add_chinese_doc('http_request.http_request.HttpResponseTooLargeError', '''\
响应体超过 max_download_bytes 时抛出的 HttpRequestError 子类，并携带 max_bytes 与已下载字节数。
''')

add_english_doc('http_request.http_request.HttpResponseTooLargeError', '''\
Subclass of HttpRequestError raised when the response body exceeds max_download_bytes,
carrying max_bytes and the downloaded byte count.
''')

add_chinese_doc('http_request.http_executor_response.HttpExecutorResponse.extract_file_info', '''\
以归一化方式识别并提取响应中的文件。

根据 Content-Disposition、内容类型或 URL 后缀判断是否为文件，并从 Content-Disposition
响应头或 URL 路径推导文件名。

**Returns:**\n
- dict 或 None: 文件响应返回 {'filename': 文件名, 'content_type': 内容类型,
  'content': 二进制内容}，非文件响应返回 None。
''')

add_english_doc('http_request.http_executor_response.HttpExecutorResponse.extract_file_info', '''\
Detect and extract a file from the response in normalized form.

A response is treated as a file based on Content-Disposition, content type or URL suffix;
the filename is derived from the Content-Disposition header or the URL path.

**Returns:**\n
- dict or None: {'filename', 'content_type', 'content'} for file responses, otherwise None.
''')

