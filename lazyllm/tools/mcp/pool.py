import asyncio
import concurrent.futures
import contextlib
import importlib.util
import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Optional

import anyio

from lazyllm import LOG
from lazyllm.thirdparty import mcp

# JSON-RPC / MCP numeric error codes. Stable across mcp 1.x and 2.x.
CONNECTION_CLOSED = -32000
REQUEST_TIMEOUT = -32001
INTERNAL_ERROR = -32603
PARSE_ERROR = -32700
_RETRYABLE_MCP_CODES = frozenset({CONNECTION_CLOSED, REQUEST_TIMEOUT, INTERNAL_ERROR})


class MCPToolError(Exception):
    def __init__(self, *, source: 'ServerSource', operation: str, attempts: int,
                 last_error: Optional[BaseException]):
        self.source = source
        self.operation = operation
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(self.describe())

    def describe(self) -> str:
        error = 'no error captured' if self.last_error is None else (
            f'{type(self.last_error).__name__}: {self.last_error}'
        )
        return (
            f'MCP {self.operation} failed after {self.attempts} attempt(s); '
            f'source: {self.source.describe()}; last error: {error}'
        )

    def __str__(self) -> str:
        return self.describe()

    def to_call_tool_result(self, tool_name: str = ''):
        return mcp.types.CallToolResult(
            content=[mcp.types.TextContent(type='text', text=self.describe())],
            isError=True,
        )


@dataclass(frozen=True)
class ServerSource:
    server_id: str
    transport: str
    target: str
    args: tuple = ()

    def describe(self) -> str:
        args = f', args={list(self.args)}' if self.args else ''
        return (
            f'server_id={self.server_id or "-"}, transport={self.transport}, '
            f'target={self.target}{args}'
        )


@dataclass(frozen=True)
class PoolKey:
    server_id: str
    transport: str
    target: str
    args: tuple
    env_fpr: str
    headers_fpr: str

    @property
    def fingerprint(self) -> str:
        return '|'.join([
            self.server_id, self.transport, self.target,
            json.dumps(list(self.args)), self.env_fpr, self.headers_fpr,
        ])


@dataclass
class PoolOptions:
    max_connections: int = 8
    max_idle_time: float = 300
    health_check_interval: float = 30
    health_check_timeout: float = 3
    acquire_timeout: float = 30
    max_retries: int = 2
    backoff_base: float = 0.5
    backoff_max: float = 10


@dataclass
class _Command:
    name: str
    args: tuple
    kwargs: dict
    timeout: Optional[float]
    reply_send: Any
    cancel_scope: anyio.CancelScope


def _stable_dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def make_pool_key(*, server_id: str, transport: str, target: str, args: list,
                  env: Optional[dict], headers: Optional[dict]) -> PoolKey:
    key = PoolKey(
        server_id=server_id or '',
        transport=transport,
        target=target,
        args=tuple(args or []),
        env_fpr=_stable_dump(env or {}),
        headers_fpr=_stable_dump(headers or {}),
    )
    return key


async def connect_session(*, source: ServerSource, env: Optional[dict],
                          headers: Optional[dict], client_timeout: float):
    stack = contextlib.AsyncExitStack()
    transport = source.transport

    if transport == 'stdio':
        params = mcp.StdioServerParameters(
            command=source.target, args=list(source.args), env=env
        )
        read_stream, write_stream = await stack.enter_async_context(mcp.stdio_client(params))
    elif transport == 'streamable-http':
        streamable_http_module = _load_mcp_module('mcp.client.streamable_http')
        create_mcp_http_client = streamable_http_module.create_mcp_http_client
        # Let the MCP SDK build the timeout object: some releases vendor httpx
        # as httpx2 and cannot consume an httpx Timeout from our shim.
        http_client = await stack.enter_async_context(
            create_mcp_http_client(headers=headers or None, timeout=client_timeout)
        )
        streams = await stack.enter_async_context(
            streamable_http_module.streamable_http_client(url=source.target, http_client=http_client)
        )
        read_stream, write_stream = streams[:2]
    else:  # sse
        sse_module = _load_mcp_module('mcp.client.sse')
        read_stream, write_stream = await stack.enter_async_context(
            sse_module.sse_client(url=source.target, headers=headers, timeout=client_timeout)
        )

    session = mcp.ClientSession(read_stream, write_stream)
    await stack.enter_async_context(session)
    await session.initialize()
    return stack, session


def _load_mcp_module(dotted: str):
    spec = importlib.util.find_spec(dotted)
    if spec is None:
        raise ImportError(
            'Please install mcp to use mcp module. '
            'You can install it with `pip install mcp`'
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SessionActor:
    def __init__(self, pool: 'ServerPool'):
        self._pool = pool
        self.stack = None
        self.session = None
        self.last_used = time.monotonic()
        self.broken = False
        self._ready = asyncio.Event()
        self._open_error: Optional[BaseException] = None
        self._cmd_send, self._cmd_receive = anyio.create_memory_object_stream(1)
        self._task: Optional[asyncio.Task] = None

    def start(self) -> asyncio.Task:
        self._task = asyncio.create_task(self._run())
        return self._task

    async def wait_ready(self):
        await self._ready.wait()
        return self._open_error

    async def _run(self):
        try:
            self.stack, self.session = await connect_session(
                source=self._pool.source,
                env=self._pool.env,
                headers=self._pool.headers,
                client_timeout=self._pool.client_timeout,
            )
        except BaseException as exc:
            self._open_error = exc
            self.broken = True
            self._ready.set()
            return

        self._ready.set()
        try:
            async with self._cmd_receive:
                async for command in self._cmd_receive:
                    await self._handle(command)
                    if self.broken:
                        break
        finally:
            await self._close_stack()
            self._pool._on_actor_stopped(self)

    async def _handle(self, command: _Command):
        with command.cancel_scope:
            try:
                coro = self._dispatch(command)
                if command.timeout is not None:
                    coro = asyncio.wait_for(coro, command.timeout)
                result = await coro
            except asyncio.TimeoutError as exc:
                self.broken = True
                await self._reply(command, ('broken', exc))
            except asyncio.CancelledError as exc:
                self.broken = True
                await self._reply(command, ('broken', exc))
            except BaseException as exc:
                kind = 'broken' if _is_connection_fault(exc) else 'error'
                if kind == 'broken':
                    self.broken = True
                await self._reply(command, (kind, exc))
            else:
                await self._reply(command, ('ok', result))

    def _dispatch(self, command: _Command) -> Coroutine:
        if command.name == 'call':
            tool_name, arguments = command.args
            return self.session.call_tool(tool_name, arguments)
        if command.name == 'list':
            return self.session.list_tools(**command.kwargs)
        if command.name == 'ping':
            return self.session.send_ping()
        if command.name == 'close':
            self.broken = True
            close_future: asyncio.Future = asyncio.get_running_loop().create_future()
            close_future.set_result(None)
            return close_future
        raise ValueError(f'unknown command {command.name}')

    async def _reply(self, command: _Command, payload):
        with anyio.CancelScope(shield=True):
            try:
                await command.reply_send.send(payload)
            except (anyio.BrokenResourceError, anyio.ClosedResourceError):
                pass

    async def invoke(self, name: str, args: tuple = (), kwargs: Optional[dict] = None,
                     timeout: Optional[float] = None):
        reply_send, reply_receive = anyio.create_memory_object_stream(1)
        command = _Command(
            name=name,
            args=args,
            kwargs=kwargs or {},
            timeout=timeout,
            reply_send=reply_send,
            cancel_scope=anyio.CancelScope(),
        )
        await self._cmd_send.send(command)
        try:
            return await reply_receive.receive()
        except asyncio.CancelledError:
            command.cancel_scope.cancel()
            with anyio.CancelScope(shield=True):
                with contextlib.suppress(BaseException):
                    await reply_receive.receive()
            raise

    async def request_close(self):
        with contextlib.suppress(anyio.BrokenResourceError, anyio.ClosedResourceError):
            reply_send, reply_receive = anyio.create_memory_object_stream(1)
            command = _Command(
                name='close', args=(), kwargs={}, timeout=None,
                reply_send=reply_send, cancel_scope=anyio.CancelScope(),
            )
            await self._cmd_send.send(command)

    async def _close_stack(self):
        with anyio.CancelScope(shield=True):
            if self.stack is not None:
                with contextlib.suppress(BaseException):
                    await self.stack.aclose()
                self.stack = None


class ServerPool:
    def __init__(self, key: PoolKey, *, env: Optional[dict], headers: Optional[dict],
                 client_timeout: float, options: PoolOptions):
        self.key = key
        self.source = ServerSource(
            server_id=key.server_id, transport=key.transport,
            target=key.target, args=key.args,
        )
        self.env = env
        self.headers = headers
        self.client_timeout = client_timeout
        self.options = options
        self._idle: list[SessionActor] = []
        self._all: set[SessionActor] = set()
        self._waiters: list[asyncio.Future] = []
        self._lock = asyncio.Lock()

    def stats(self) -> dict:
        return {
            'total': len(self._all),
            'idle': len(self._idle),
            'in_use': len(self._all) - len(self._idle),
            'waiters': len(self._waiters),
        }

    async def acquire(self) -> SessionActor:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.options.acquire_timeout
        while True:
            async with self._lock:
                decision = self._pick_or_queue(loop)

            if decision[0] == 'wait':
                await self._await_release(decision[1], deadline)
                continue

            actor = decision[1]
            if decision[0] == 'idle':
                if await self._use_or_discard_idle(actor):
                    return actor
                continue

            await self._await_fresh_actor(actor)
            return actor

    def _pick_or_queue(self, loop) -> tuple:
        if self._idle:
            return ('idle', self._idle.pop())
        if len(self._all) < self.options.max_connections:
            actor = SessionActor(self)
            self._all.add(actor)
            actor.start()
            return ('fresh', actor)
        waiter_future = loop.create_future()
        self._waiters.append(waiter_future)
        return ('wait', waiter_future)

    async def _await_release(self, waiter_future, deadline):
        loop = asyncio.get_running_loop()
        try:
            await asyncio.wait_for(
                asyncio.shield(waiter_future), deadline - loop.time()
            )
        except asyncio.TimeoutError:
            async with self._lock:
                if not waiter_future.done():
                    self._waiters.remove(waiter_future)
            raise

    async def _use_or_discard_idle(self, actor) -> bool:
        if time.monotonic() - actor.last_used > self.options.max_idle_time:
            await self._discard(actor)
            return False
        healthy = await self._health_check(actor)
        return bool(healthy)

    async def _await_fresh_actor(self, actor):
        try:
            error = await actor.wait_ready()
        except asyncio.CancelledError:
            await self._cancel_fresh_actor(actor)
            raise
        if error is not None:
            async with self._lock:
                self._all.discard(actor)
            raise error

    async def _cancel_fresh_actor(self, actor: SessionActor):
        actor._task.cancel()
        with anyio.CancelScope(shield=True):
            with contextlib.suppress(BaseException):
                await actor._task
        async with self._lock:
            self._all.discard(actor)

    async def _health_check(self, actor: SessionActor) -> bool:
        interval = self.options.health_check_interval
        if interval is None or time.monotonic() - actor.last_used < interval:
            return True
        try:
            kind, _payload = await actor.invoke(
                'ping', timeout=self.options.health_check_timeout
            )
        except BaseException:
            await self._discard(actor)
            return False
        if kind != 'ok' or actor.broken:
            await self._discard(actor)
            return False
        return True

    async def _discard(self, actor: SessionActor):
        async with self._lock:
            await self._discard_locked(actor)

    async def release(self, actor: SessionActor, broken: bool):
        async with self._lock:
            if broken or actor.broken:
                await self._discard_locked(actor)
                if not actor.broken:
                    await actor.request_close()
            else:
                actor.last_used = time.monotonic()
                self._idle.append(actor)
                if self._waiters:
                    waiter = self._waiters.pop(0)
                    if not waiter.done():
                        waiter.set_result(None)

    async def _discard_locked(self, actor: SessionActor):
        self._idle = [a for a in self._idle if a is not actor]
        self._all.discard(actor)
        if not actor.broken:
            await actor.request_close()

    def _on_actor_stopped(self, actor: SessionActor):
        asyncio.create_task(self._actor_stopped(actor))

    async def _actor_stopped(self, actor: SessionActor):
        async with self._lock:
            self._idle = [a for a in self._idle if a is not actor]
            existed = actor in self._all
            self._all.discard(actor)
            if self._waiters:
                waiter = self._waiters.pop(0)
                if not waiter.done():
                    waiter.set_result(None)
        if existed:
            LOG.debug(f'MCP session actor stopped; {self.source.describe()}')

    async def evict_expired_idle(self) -> int:
        removed = 0
        async with self._lock:
            expired = [
                actor for actor in self._idle
                if time.monotonic() - actor.last_used > self.options.max_idle_time
            ]
            for actor in expired:
                await self._discard_locked(actor)
                removed += 1
        return removed

    async def aclose(self):
        async with self._lock:
            actors = list(self._all) + list(self._idle)
            self._idle.clear()
            self._all.clear()
            for waiter in self._waiters:
                if not waiter.done():
                    waiter.set_exception(MCPToolError(
                        source=self.source, operation='acquire', attempts=0, last_error=None,
                    ))
            self._waiters.clear()
        for actor in actors:
            await actor.request_close()


def _is_connection_fault(exc: BaseException) -> bool:
    if isinstance(exc, BaseExceptionGroup):
        return any(_is_connection_fault(nested) for nested in exc.exceptions)

    if isinstance(exc, asyncio.TimeoutError):
        return True

    error = getattr(exc, 'error', None)
    code = getattr(error, 'code', None)
    if code is not None and code in _RETRYABLE_MCP_CODES:
        return True
    message = getattr(error, 'message', '')
    # Remote HTTP transports report an expired/unknown session with a generic
    # INVALID_REQUEST code; the message reveals the session is unusable.
    if error is not None:
        if 'Session terminated' in message or 'Session not found' in message:
            return True

    retryable_modules = ('httpx', 'httpx2', 'httpcore', 'httpcore2', 'anyio')
    for cls in type(exc).__mro__:
        module = cls.__module__ or ''
        if module.startswith(retryable_modules):
            if cls.__name__ in (
                'TransportError', 'TimeoutException', 'NetworkError', 'ConnectError',
                'ReadError', 'WriteError', 'RemoteProtocolError', 'PoolTimeout',
                'ConnectTimeout', 'ReadTimeout', 'WriteTimeout', 'BrokenResourceError',
                'ClosedResourceError', 'EndOfStream',
            ):
                return True

    if isinstance(exc, ConnectionError):
        return True
    return False


def is_retryable_error(exc: BaseException) -> bool:
    return _is_connection_fault(exc)


class _PoolDriver:
    def __init__(self):
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self._pools: dict[str, ServerPool] = {}
        self._pool_meta: dict[str, tuple] = {}
        self._reaper: Optional[asyncio.Task] = None
        self._reaper_event: Optional[asyncio.Event] = None
        self._lock = threading.Lock()
        self._closing = False

    def ensure_started(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self.loop is not None and self.thread is not None and self.thread.is_alive():
                return self.loop
            ready = threading.Event()
            self.loop = None

            def runner():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self.loop = loop
                self._reaper_event = asyncio.Event()
                self._reaper = loop.create_task(self._reap_loop())
                ready.set()
                loop.run_forever()

            self.thread = threading.Thread(
                target=runner, name='lazyllm-mcp-pool', daemon=True
            )
            self.thread.start()
            ready.wait()
            self._closing = False
            return self.loop

    def submit(self, factory: Callable[[], Coroutine]) -> concurrent.futures.Future:
        loop = self.ensure_started()
        return asyncio.run_coroutine_threadsafe(factory(), loop)

    async def _reap_loop(self):
        while True:
            await asyncio.sleep(15)
            for pool in list(self._pools.values()):
                with contextlib.suppress(Exception):
                    await pool.evict_expired_idle()

    def get_pool(self, **pool_kwargs) -> ServerPool:
        key = pool_kwargs['key']
        fingerprint = key.fingerprint
        pool = self._pools.get(fingerprint)
        if pool is None:
            pool = ServerPool(
                key,
                env=pool_kwargs['env'],
                headers=pool_kwargs['headers'],
                client_timeout=pool_kwargs['client_timeout'],
                options=pool_kwargs['options'],
            )
            self._pools[fingerprint] = pool
            self._pool_meta[fingerprint] = (
                pool_kwargs['env'], pool_kwargs['headers'],
                pool_kwargs['client_timeout'], pool_kwargs['options'],
            )
        return pool

    async def execute_call_tool(self, *, key: PoolKey, env: Optional[dict],
                                headers: Optional[dict], client_timeout: float,
                                options: PoolOptions, tool_name: str,
                                arguments: dict, call_timeout: Optional[float]):
        pool = self.get_pool(
            key=key, env=env, headers=headers,
            client_timeout=client_timeout, options=options,
        )
        last_error: Optional[BaseException] = None
        attempts = options.max_retries + 1

        for attempt in range(1, attempts + 1):
            scope = anyio.CancelScope()
            actor: Optional[SessionActor] = None
            try:
                with scope:
                    actor = await pool.acquire()
                    kind, payload = await actor.invoke(
                        'call', (tool_name, arguments), timeout=call_timeout
                    )
            except asyncio.CancelledError as exc:
                if scope.cancel_called:
                    if actor is not None:
                        with anyio.CancelScope(shield=True):
                            await pool.release(actor, broken=True)
                    raise
                # CancelledError raised from inner SDK scopes (e.g. connect
                # refused) rather than by cancelling this task.
                last_error = exc
                if actor is not None:
                    await pool.release(actor, broken=True)
            except BaseException as exc:
                last_error = exc
                if actor is not None:
                    await pool.release(actor, broken=True)
            else:
                if kind == 'ok':
                    await pool.release(actor, broken=False)
                    return payload
                last_error = payload
                evict = kind == 'broken' or is_retryable_error(payload)
                await pool.release(actor, broken=evict)
                if not is_retryable_error(payload):
                    break
            if attempt < attempts:
                delay = min(
                    options.backoff_base * (2 ** (attempt - 1)),
                    options.backoff_max,
                )
                LOG.warning(
                    f'MCP tool "{tool_name}" attempt {attempt} failed '
                    f'({type(last_error).__name__}: {last_error}); '
                    f'retrying in {delay:.2f}s; {pool.source.describe()}'
                )
                await asyncio.sleep(delay)

        LOG.error(
            f'MCP tool "{tool_name}" failed after {attempt} attempts; '
            f'{pool.source.describe()}'
        )
        failure = MCPToolError(
            source=pool.source, operation=f'call_tool[{tool_name}]',
            attempts=attempt, last_error=last_error,
        )
        return failure.to_call_tool_result(tool_name)

    async def execute_list_tools(self, *, key: PoolKey, env: Optional[dict],
                                 headers: Optional[dict], client_timeout: float,
                                 options: PoolOptions, params=None):
        pool = self.get_pool(
            key=key, env=env, headers=headers,
            client_timeout=client_timeout, options=options,
        )
        last_error: Optional[BaseException] = None
        attempts = options.max_retries + 1

        for attempt in range(1, attempts + 1):
            scope = anyio.CancelScope()
            actor: Optional[SessionActor] = None
            try:
                with scope:
                    actor = await pool.acquire()
                    kind, payload = await actor.invoke(
                        'list', kwargs={'params': params}, timeout=client_timeout
                    )
            except asyncio.CancelledError as exc:
                if scope.cancel_called:
                    if actor is not None:
                        with anyio.CancelScope(shield=True):
                            await pool.release(actor, broken=True)
                    raise
                last_error = exc
                if actor is not None:
                    await pool.release(actor, broken=True)
            except BaseException as exc:
                last_error = exc
                if actor is not None:
                    await pool.release(actor, broken=True)
            else:
                if kind == 'ok':
                    await pool.release(actor, broken=False)
                    return payload
                last_error = payload
                evict = kind == 'broken' or is_retryable_error(payload)
                await pool.release(actor, broken=evict)
                if not is_retryable_error(payload):
                    break
            if attempt < attempts:
                delay = min(
                    options.backoff_base * (2 ** (attempt - 1)),
                    options.backoff_max,
                )
                await asyncio.sleep(delay)

        raise MCPToolError(
            source=pool.source, operation='list_tools',
            attempts=attempt, last_error=last_error,
        )

    async def shutdown_all(self):
        with self._lock:
            self._closing = True
        pools = list(self._pools.values())
        for pool in pools:
            with contextlib.suppress(Exception):
                await pool.aclose()
        self._pools.clear()
        self._pool_meta.clear()
        if self._reaper is not None:
            self._reaper.cancel()
            with contextlib.suppress(BaseException):
                await self._reaper
            self._reaper = None

    def pool_stats(self) -> dict:
        return {fpr: pool.stats() for fpr, pool in self._pools.items()}


_DRIVER: Optional[_PoolDriver] = None
_DRIVER_GUARD = threading.Lock()


def get_driver() -> _PoolDriver:
    global _DRIVER
    if _DRIVER is None:
        with _DRIVER_GUARD:
            if _DRIVER is None:
                _DRIVER = _PoolDriver()
    return _DRIVER


async def aclose_all_pools():
    if _DRIVER is not None and _DRIVER.loop is not None:
        loop = _DRIVER.loop
        await asyncio.wrap_future(
            asyncio.run_coroutine_threadsafe(_DRIVER.shutdown_all(), loop)
        )


def reset_driver_for_tests():
    global _DRIVER
    driver = _DRIVER
    if driver is None:
        return
    if driver.loop is not None and driver.thread is not None and driver.thread.is_alive():
        future = asyncio.run_coroutine_threadsafe(driver.shutdown_all(), driver.loop)
        future.result(timeout=30)
        driver.loop.call_soon_threadsafe(driver.loop.stop)
        driver.thread.join(timeout=5)
    with _DRIVER_GUARD:
        _DRIVER = None
