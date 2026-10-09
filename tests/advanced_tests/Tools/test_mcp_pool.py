import asyncio
import inspect
import os
import platform
import time

import pytest

from lazyllm.tools import MCPClient
from lazyllm.tools.mcp.pool import get_driver, reset_driver_for_tests

from mcp_mock_servers import HttpServerThread, build_remote_app

HERE = os.path.dirname(os.path.abspath(__file__))
STDIO_SCRIPT = os.path.join(HERE, 'mcp_mock_stdio.py')
FLAKY_LAUNCH_SCRIPT = os.path.join(HERE, 'mcp_mock_flaky_launch.py')
IS_WINDOWS = platform.system() == 'Windows'


@pytest.fixture(autouse=True)
def _reset_pools():
    reset_driver_for_tests()
    yield
    reset_driver_for_tests()


def stdio_client_factory(**kwargs):
    import sys
    defaults = dict(
        command_or_url=sys.executable,
        args=[STDIO_SCRIPT],
        timeout=15,
        max_retries=1,
        retry_backoff_base=0.05,
    )
    defaults.update(kwargs)
    return MCPClient(**defaults)


def result_is_error(result) -> bool:
    return bool(getattr(result, 'is_error', getattr(result, 'isError', False)))


def result_text(result) -> str:
    return result.content[0].text


async def wait_stats_settle(expected_total=None, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        stats = list(get_driver().pool_stats().values())
        if not stats:
            if expected_total in (None, 0):
                return []
        elif expected_total is None or all(s['total'] == expected_total for s in stats):
            return stats
        await asyncio.sleep(0.02)
    raise AssertionError(f'stats did not settle to total={expected_total}: {get_driver().pool_stats()}')


# ---------------------------------------------------------------- stdio

class TestStdio:
    def test_get_tools_and_schema(self):
        client = stdio_client_factory()
        tools = client.get_tools()
        names = [t.__name__ for t in tools]
        assert 'echo' in names and 'slow' in names

        echo_tool = next(t for t in tools if t.__name__ == 'echo')
        sig = inspect.signature(echo_tool)
        param = sig.parameters['text']
        assert param.kind == inspect.Parameter.KEYWORD_ONLY
        assert param.annotation is str
        assert 'Args:' in (echo_tool.__doc__ or '')
        assert echo_tool.__mcp_tool_name__ == 'echo'

        filtered = client.get_tools(allowed_tools=['echo'])
        assert len(filtered) == 1 and filtered[0].__name__ == 'echo'

    def test_sync_tool_call(self):
        client = stdio_client_factory()
        tool = client.get_tools(allowed_tools=['echo'])[0]
        res = tool(text='sync-hello')
        assert 'Received text message:' in res and 'sync-hello' in res

    @pytest.mark.asyncio
    async def test_connection_reuse(self):
        client = stdio_client_factory()
        r1 = await client.call_tool('echo', {'text': 'a'})
        assert result_text(r1) == 'a'
        r2 = await client.call_tool('echo', {'text': 'b'})
        assert result_text(r2) == 'b'

        stats = list(get_driver().pool_stats().values())
        assert len(stats) == 1
        assert stats[0]['total'] == 1 and stats[0]['idle'] == 1

    @pytest.mark.asyncio
    async def test_timeout(self):
        client = stdio_client_factory(max_retries=0)
        result = await client.call_tool('slow', {'delay': 5}, timeout=0.5)
        assert result_is_error(result)
        text = result_text(result)
        assert 'transport=stdio' in text
        assert 'TimeoutError' in text
        assert 'call_tool[slow]' in text
        await wait_stats_settle(expected_total=0)

    @pytest.mark.asyncio
    async def test_retry_then_success_launch(self, tmp_path):
        count_file = tmp_path / 'launch.count'
        import sys
        client = MCPClient(
            command_or_url=sys.executable,
            args=[FLAKY_LAUNCH_SCRIPT],
            env={**os.environ,
                 'MCP_LAUNCH_COUNT_FILE': str(count_file),
                 'MCP_LAUNCH_FAIL_N': '1'},
            timeout=15,
            max_retries=1,
            retry_backoff_base=0.05,
        )
        result = await client.call_tool('echo', {'text': 'after-retry'})
        assert not result_is_error(result)
        assert result_text(result) == 'after-retry'

    @pytest.mark.asyncio
    async def test_cancel(self):
        client = stdio_client_factory(max_retries=0)
        task = asyncio.create_task(client.call_tool('slow', {'delay': 5}, timeout=10))
        await asyncio.sleep(0.5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await wait_stats_settle(expected_total=0)

        result = await client.call_tool('echo', {'text': 'recovered'})
        assert result_text(result) == 'recovered'

    @pytest.mark.asyncio
    async def test_dead_session_eviction(self):
        client = stdio_client_factory()
        warm = await client.call_tool('echo', {'text': 'warm'})
        assert result_text(warm) == 'warm'

        result = await client.call_tool('crash', {}, timeout=10)
        assert result_is_error(result)
        text = result_text(result)
        assert 'Connection closed' in text
        assert 'transport=stdio' in text

        await wait_stats_settle(expected_total=0)

        recovered = await client.call_tool('echo', {'text': 'fresh'})
        assert result_text(recovered) == 'fresh'

    @pytest.mark.asyncio
    async def test_application_error_does_not_evict(self):
        client = stdio_client_factory(max_retries=0)
        result = await client.call_tool('app_error', {'text': 'boom'})
        assert result_is_error(result)
        stats = list(get_driver().pool_stats().values())
        assert stats[0]['total'] == 1 and stats[0]['idle'] == 1

        follow = await client.call_tool('echo', {'text': 'still-here'})
        assert result_text(follow) == 'still-here'
        assert list(get_driver().pool_stats().values())[0]['total'] == 1

    @pytest.mark.asyncio
    async def test_concurrency_isolation(self):
        client = stdio_client_factory(max_connections=6)
        # Pre-spawn five actors so subprocess startup does not mask whether
        # tool calls themselves run concurrently.
        warmed = await asyncio.gather(*[
            client.call_tool('echo', {'text': str(i)}) for i in range(5)
        ])
        assert [result_text(r) for r in warmed] == [str(i) for i in range(5)]
        stats = list(get_driver().pool_stats().values())[0]
        assert stats['total'] == 5 and stats['idle'] == 5

        start = time.monotonic()
        results = await asyncio.gather(*[
            client.call_tool('slow', {'delay': 0.5}) for _ in range(5)
        ])
        elapsed = time.monotonic() - start
        assert all(result_text(r) == 'done' for r in results)
        assert elapsed < 1.5, f'calls were serialized: {elapsed:.2f}s'
        stats = list(get_driver().pool_stats().values())[0]
        assert stats['total'] == 5 and stats['idle'] == 5


# ---------------------------------------------------------------- remote transports

REMOTE_CASES = [
    pytest.param('sse', '/sse', id='sse'),
    pytest.param('streamable-http', '/mcp', id='streamable-http'),
]


@pytest.mark.parametrize('transport,path', REMOTE_CASES)
class TestRemoteTransports:
    def test_get_tools_and_call(self, transport, path):
        app, _, _ = build_remote_app(transport)
        with HttpServerThread(app) as server:
            client = MCPClient(
                f'http://127.0.0.1:{server.port}{path}',
                timeout=15, max_retries=1, retry_backoff_base=0.05,
                transport=transport,
            )
            tools = client.get_tools(allowed_tools=['echo'])
            assert len(tools) == 1 and tools[0].__name__ == 'echo'
            res = tools[0](text='remote-hello')
            assert 'remote-hello' in res

    def test_connection_reuse(self, transport, path):
        app, _, _ = build_remote_app(transport)
        with HttpServerThread(app) as server:
            client = MCPClient(
                f'http://127.0.0.1:{server.port}{path}',
                timeout=15, transport=transport,
            )

            async def scenario():
                r1 = await client.call_tool('echo', {'text': 'a'})
                r2 = await client.call_tool('echo', {'text': 'b'})
                assert result_text(r1) == 'a' and result_text(r2) == 'b'
                stats = list(get_driver().pool_stats().values())
                assert stats[0]['total'] == 1

            asyncio.run(scenario())

    def test_retry_then_success(self, transport, path):
        app, _, _ = build_remote_app(transport, http_fail=2)
        with HttpServerThread(app) as server:
            client = MCPClient(
                f'http://127.0.0.1:{server.port}{path}',
                timeout=15, transport=transport,
                max_retries=2, retry_backoff_base=0.05,
            )

            async def scenario():
                result = await client.call_tool('echo', {'text': 'third-time'})
                assert not result_is_error(result)
                assert result_text(result) == 'third-time'

            asyncio.run(scenario())

    def test_connect_refused(self, transport, path):
        client = MCPClient(
            f'http://127.0.0.1:16543{path}',
            timeout=2, transport=transport,
            max_retries=1, retry_backoff_base=0.05,
        )

        async def scenario():
            result = await client.call_tool('echo', {'text': 'x'})
            assert result_is_error(result)
            text = result_text(result)
            assert f'transport={transport}' in text
            assert '16543' in text

        asyncio.run(scenario())

    def test_dead_session_evicted_and_restarted(self, transport, path):
        port_holder = {}
        app, _, _ = build_remote_app(transport)
        server = HttpServerThread(app).start()
        port_holder['port'] = server.port

        async def scenario():
            client = MCPClient(
                f'http://127.0.0.1:{port_holder["port"]}{path}',
                timeout=10, transport=transport,
                max_retries=2, retry_backoff_base=0.2,
            )
            warm = await client.call_tool('echo', {'text': 'warm'})
            assert result_text(warm) == 'warm'

            server.stop()

            def restart():
                new_app, _, _ = build_remote_app(transport)
                new_server = HttpServerThread(
                    new_app, port=port_holder['port']
                ).start()
                port_holder['new_server'] = new_server

            await asyncio.get_running_loop().run_in_executor(None, restart)

            # Pooled actor belongs to the dead server; this call must evict
            # it and complete against the restarted server on a fresh actor.
            result = await client.call_tool('echo', {'text': 'reborn'})
            assert not result_is_error(result), result_text(result)
            assert result_text(result) == 'reborn'

        asyncio.run(scenario())
        new_server = port_holder.get('new_server')
        if new_server is not None:
            new_server.stop()


class TestStreamableSpecifics:
    def test_timeout_over_http(self):
        app, _, _ = build_remote_app('streamable-http')
        with HttpServerThread(app) as server:
            client = MCPClient(
                f'http://127.0.0.1:{server.port}/mcp',
                timeout=15, transport='streamable-http', max_retries=0,
            )

            async def scenario():
                result = await client.call_tool('slow', {'delay': 5}, timeout=0.5)
                assert result_is_error(result)
                text = result_text(result)
                assert 'transport=streamable-http' in text
                assert 'TimeoutError' in text

            asyncio.run(scenario())

    def test_cancel_over_http(self):
        app, _, _ = build_remote_app('streamable-http')
        server = HttpServerThread(app).start()

        async def scenario():
            client = MCPClient(
                f'http://127.0.0.1:{server.port}/mcp',
                timeout=15, transport='streamable-http', max_retries=0,
            )
            task = asyncio.create_task(
                client.call_tool('slow', {'delay': 5}, timeout=10)
            )
            await asyncio.sleep(0.5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            recovered = await client.call_tool('echo', {'text': 'back'})
            assert result_text(recovered) == 'back'

        asyncio.run(scenario())
        server.stop()

    def test_health_check_eviction(self):
        port_holder = {}
        app, _, _ = build_remote_app('streamable-http')
        server = HttpServerThread(app).start()
        port_holder['port'] = server.port

        async def scenario():
            client = MCPClient(
                f'http://127.0.0.1:{port_holder["port"]}/mcp',
                timeout=10, transport='streamable-http',
                health_check_interval=0.2,
                max_retries=1, retry_backoff_base=0.1,
            )
            warm = await client.call_tool('echo', {'text': 'warm'})
            assert result_text(warm) == 'warm'
            assert list(get_driver().pool_stats().values())[0]['idle'] == 1

            server.stop()
            await asyncio.sleep(0.5)

            new_app, _, _ = build_remote_app('streamable-http')
            new_server = HttpServerThread(new_app, port=port_holder['port']).start()
            port_holder['new_server'] = new_server

            result = await client.call_tool('echo', {'text': 'healthy-again'})
            assert not result_is_error(result), result_text(result)
            assert result_text(result) == 'healthy-again'

        asyncio.run(scenario())
        new_server = port_holder.get('new_server')
        if new_server is not None:
            new_server.stop()

    def test_concurrency_isolation_over_http(self):
        app, _, _ = build_remote_app('streamable-http')
        server = HttpServerThread(app).start()

        async def scenario():
            client = MCPClient(
                f'http://127.0.0.1:{server.port}/mcp',
                timeout=15, transport='streamable-http',
                max_connections=6,
            )
            start = time.monotonic()
            results = await asyncio.gather(*[
                client.call_tool('slow', {'delay': 0.5}) for _ in range(5)
            ])
            elapsed = time.monotonic() - start
            assert all(result_text(r) == 'done' for r in results)
            assert elapsed < 1.5
            stats = list(get_driver().pool_stats().values())[0]
            assert stats['total'] == 5

        asyncio.run(scenario())
        server.stop()


class TestPoolIsolation:
    def test_different_servers_do_not_share(self):
        app_a, _, _ = build_remote_app('streamable-http')
        app_b, _, _ = build_remote_app('streamable-http')
        server_a = HttpServerThread(app_a).start()
        server_b = HttpServerThread(app_b).start()

        async def scenario():
            client_a = MCPClient(
                f'http://127.0.0.1:{server_a.port}/mcp',
                timeout=10, transport='streamable-http', server_id='A',
            )
            client_b = MCPClient(
                f'http://127.0.0.1:{server_b.port}/mcp',
                timeout=10, transport='streamable-http', server_id='B',
            )
            await client_a.call_tool('echo', {'text': 'a'})
            await client_b.call_tool('echo', {'text': 'b'})

            stats = get_driver().pool_stats()
            assert len(stats) == 2
            assert all(s['total'] == 1 for s in stats.values())

            # Same config -> shared pool actor.
            client_a2 = MCPClient(
                f'http://127.0.0.1:{server_a.port}/mcp',
                timeout=10, transport='streamable-http', server_id='A',
            )
            await client_a2.call_tool('echo', {'text': 'a2'})
            assert len(get_driver().pool_stats()) == 2
            assert sum(s['total'] for s in get_driver().pool_stats().values()) == 2

        asyncio.run(scenario())
        server_a.stop()
        server_b.stop()

    def test_different_headers_env_separate_pools(self):
        client_a = stdio_client_factory(env={**os.environ, 'POOL_TAG': 'x'})
        client_b = stdio_client_factory(env={**os.environ, 'POOL_TAG': 'y'})

        async def scenario():
            await client_a.call_tool('echo', {'text': 'a'})
            await client_b.call_tool('echo', {'text': 'b'})
            assert len(get_driver().pool_stats()) == 2

        asyncio.run(scenario())
