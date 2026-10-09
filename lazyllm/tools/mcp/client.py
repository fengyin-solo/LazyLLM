import asyncio
import contextlib
from typing import Any, Optional, Literal
from urllib.parse import urlparse

from .pool import (
    PoolOptions,
    ServerSource,
    connect_session,
    get_driver,
    make_pool_key,
)
from .utils import patch_sync
from .tool_adaptor import generate_lazyllm_tool
from .deploy import SseServerSettings, start_sse_server


class MCPClient(object):
    def __init__(
        self,
        command_or_url: str,
        args: Optional[list[str]] = None,
        env: dict[str, str] = None,
        headers: dict[str, Any] = None,
        timeout: float = 5,
        transport: Literal['auto', 'stdio', 'sse', 'streamable-http'] = 'auto',
        server_id: str = '',
        max_connections: int = 8,
        max_idle_time: float = 300,
        health_check_interval: float = 30,
        max_retries: int = 2,
        retry_backoff_base: float = 0.5,
        retry_backoff_max: float = 10,
    ):
        self._command_or_url = command_or_url
        self._args = args or []
        self._env = env
        self._headers = headers
        self._timeout = timeout
        self._transport = transport
        self.server_id = server_id
        self._max_connections = max_connections
        self._max_idle_time = max_idle_time
        self._health_check_interval = health_check_interval
        self._max_retries = max_retries
        self._retry_backoff_base = retry_backoff_base
        self._retry_backoff_max = retry_backoff_max

    def _resolve_transport(self) -> str:
        if self._transport != 'auto':
            return self._transport
        if urlparse(self._command_or_url).scheme in ('http', 'https'):
            return 'streamable-http'
        return 'stdio'

    def _pool_key(self):
        return make_pool_key(
            server_id=self.server_id,
            transport=self._resolve_transport(),
            target=self._command_or_url,
            args=self._args,
            env=self._env,
            headers=self._headers,
        )

    def _pool_options(self) -> PoolOptions:
        return PoolOptions(
            max_connections=self._max_connections,
            max_idle_time=self._max_idle_time,
            health_check_interval=self._health_check_interval,
            max_retries=self._max_retries,
            backoff_base=self._retry_backoff_base,
            backoff_max=self._retry_backoff_max,
        )

    def _submit_to_driver(self, factory):
        driver = get_driver()
        loop = driver.ensure_started()
        return asyncio.run_coroutine_threadsafe(factory(), loop)

    async def call_tool(self, tool_name: str, arguments: dict,
                        timeout: Optional[float] = None):
        call_timeout = timeout if timeout is not None else self._timeout
        key = self._pool_key()
        options = self._pool_options()
        future = self._submit_to_driver(lambda: get_driver().execute_call_tool(
            key=key,
            env=self._env,
            headers=self._headers,
            client_timeout=self._timeout,
            options=options,
            tool_name=tool_name,
            arguments=arguments,
            call_timeout=call_timeout,
        ))
        return await asyncio.wrap_future(future)

    async def list_tools(self):
        key = self._pool_key()
        options = self._pool_options()
        future = self._submit_to_driver(lambda: get_driver().execute_list_tools(
            key=key,
            env=self._env,
            headers=self._headers,
            client_timeout=self._timeout,
            options=options,
        ))
        return await asyncio.wrap_future(future)

    async def aget_tools(self, allowed_tools: list[str] = None):
        res = await self.list_tools()
        mcp_tools = getattr(res, 'tools', [])
        if allowed_tools:
            mcp_tools = [tool for tool in mcp_tools if tool.name in allowed_tools]

        return [generate_lazyllm_tool(self, tool) for tool in mcp_tools]

    def get_tools(self, allowed_tools: list[str] = None):
        return patch_sync(self.aget_tools)(allowed_tools=allowed_tools)

    async def deploy(self, sse_settings: SseServerSettings):
        source = ServerSource(
            server_id=self.server_id,
            transport=self._resolve_transport(),
            target=self._command_or_url,
            args=tuple(self._args),
        )
        stack, session = await connect_session(
            source=source,
            env=self._env,
            headers=self._headers,
            client_timeout=self._timeout,
        )
        try:
            await start_sse_server(session, sse_settings)
        finally:
            with contextlib.suppress(Exception):
                await stack.aclose()
