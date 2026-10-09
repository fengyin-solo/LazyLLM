import json
import mimetypes
import os
import re
from urllib.parse import unquote, urlparse

from lazyllm.thirdparty import httpx

_FILE_CONTENT_TYPE_PREFIXES = ('image/', 'audio/', 'video/', 'font/')
_FILE_APPLICATION_SUBTYPES = frozenset({
    'octet-stream', 'pdf', 'zip', 'x-tar', 'gzip', 'x-gzip', 'x-bzip2', 'x-7z-compressed',
    'rar', 'x-rar-compressed',
    'msword', 'vnd.ms-excel', 'vnd.ms-powerpoint',
    'vnd.openxmlformats-officedocument.wordprocessingml.document',
    'vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'vnd.openxmlformats-officedocument.presentationml.presentation',
})
_BINARY_URL_SUFFIXES = frozenset({
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.ico', '.svg', '.tiff',
    '.pdf', '.zip', '.tar', '.gz', '.bz2', '.7z', '.rar',
    '.mp3', '.wav', '.flac', '.aac', '.ogg',
    '.mp4', '.mov', '.avi', '.mkv', '.webm',
    '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
})
_FILENAME_STAR_RE = re.compile(r"filename\*\s*=\s*([^']+)'([^']*)'([^;]+)", re.IGNORECASE)
_FILENAME_RE = re.compile(r'filename\s*=\s*"?([^";]+)"?', re.IGNORECASE)


class HttpExecutorResponse:
    headers: dict[str, str]
    response: 'httpx.Response'

    def __init__(self, response: 'httpx.Response' = None):
        self.response = response
        self.headers = dict(response.headers) if isinstance(self.response, httpx.Response) else {}

    def _require_response(self):
        if not isinstance(self.response, httpx.Response):
            raise ValueError(f'Invalid response type {type(self.response)}')
        return self.response

    @property
    def is_file(self) -> bool:
        '''
        check if response is file
        '''
        content_type = self.get_content_type()
        file_content_types = ['image', 'audio', 'video']

        return any(v in content_type for v in file_content_types)

    def get_content_type(self) -> str:
        return self.headers.get('content-type', '')

    @property
    def content_type(self) -> str:
        return self.get_content_type().split(';', 1)[0].strip().lower()

    def extract_file(self) -> tuple[str, bytes]:
        '''
        extract file from response if content type is file related
        '''
        if self.is_file:
            return self.get_content_type(), self.body

        return '', b''

    @property
    def content(self) -> str:
        return self._require_response().text

    @property
    def text(self) -> str:
        return self._require_response().text

    @property
    def body(self) -> bytes:
        return self._require_response().content

    @property
    def status_code(self) -> int:
        return self._require_response().status_code

    def json(self):
        media_type = self.content_type
        if 'json' not in media_type or not self.body:
            return None
        try:
            return self._require_response().json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    @staticmethod
    def _parse_content_disposition(header_value: str) -> tuple[bool, str]:
        is_attachment = False
        filename = ''
        if not header_value:
            return is_attachment, filename

        disposition = header_value.split(';', 1)[0].strip().lower()
        is_attachment = disposition == 'attachment'

        match = _FILENAME_STAR_RE.search(header_value)
        if match:
            filename = unquote(match.group(3).strip().strip('"'))
        else:
            match = _FILENAME_RE.search(header_value)
            if match:
                candidate = match.group(1).strip()
                if candidate and candidate.lower() != 'attachment':
                    filename = candidate
        return is_attachment or bool(filename), filename

    @staticmethod
    def _filename_from_url(url: str) -> str:
        if not url:
            return ''
        path = unquote(urlparse(url).path)
        name = os.path.basename(path)
        return name if name and os.path.splitext(name)[1].lower() in _BINARY_URL_SUFFIXES else ''

    @staticmethod
    def _is_file_media_type(media_type: str) -> bool:
        if not media_type:
            return False
        if media_type.startswith(_FILE_CONTENT_TYPE_PREFIXES):
            return True
        if media_type.startswith('application/'):
            subtype = media_type.split('/', 1)[1]
            return subtype in _FILE_APPLICATION_SUBTYPES
        return False

    @staticmethod
    def _default_filename(media_type: str) -> str:
        suffix = mimetypes.guess_extension(media_type) if media_type else None
        return f'download{suffix}' if suffix else 'download'

    def extract_file_info(self, url: str = '') -> dict:
        response = self._require_response()
        media_type = self.content_type
        disposition_file, disposition_name = self._parse_content_disposition(
            self.headers.get('content-disposition', ''))
        url_name = self._filename_from_url(url)
        filename = disposition_name or url_name

        is_file = (disposition_file or self._is_file_media_type(media_type)
                   or bool(url_name))
        if not is_file:
            return None

        if not filename:
            filename = self._default_filename(media_type)
        return {'filename': filename, 'content_type': self.get_content_type(),
                'content': response.content}
