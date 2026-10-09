import os
import threading
import time

import uvicorn
from starlette.responses import PlainTextResponse


def mcp_major_version():
    try:
        from importlib.metadata import version
        return int(version('mcp').split('.')[0])
    except Exception:
        import mcp
        return 2 if hasattr(mcp.server, 'mcpserver') else 1


def create_server(name='mcp-mock'):
    if mcp_major_version() >= 2:
        from mcp.server.mcpserver import MCPServer
        return MCPServer(name)
    from mcp.server.fastmcp import FastMCP
    return FastMCP(name)


def register_mock_tools(server, state):
    import asyncio

    @server.tool()
    def echo(text: str) -> str:
        return text

    @server.tool()
    async def slow(delay: float) -> str:
        await asyncio.sleep(delay)
        return 'done'

    @server.tool()
    def app_error(text: str) -> str:
        raise RuntimeError(f'application failure: {text}')

    @server.tool()
    def call_count() -> int:
        state['call_count'] = state.get('call_count', 0) + 1
        return state['call_count']

    @server.tool()
    def crash() -> str:
        os._exit(0)


class FlakyHTTPMiddleware:
    def __init__(self, app, state, fail_status=503, methods=('POST',)):
        self.app = app
        self.state = state
        self.fail_status = fail_status
        self.methods = methods

    async def __call__(self, scope, receive, send):
        if scope['type'] == 'http' and scope['method'] in self.methods:
            self.state['http_calls'] = self.state.get('http_calls', 0) + 1
            if self.state['http_calls'] <= self.state.get('http_fail', 0):
                response = PlainTextResponse(
                    'temporary server failure', status_code=self.fail_status
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def build_remote_app(transport, state=None, http_fail=0, fail_status=503):
    state = state if state is not None else {}
    state['http_fail'] = http_fail
    server = create_server()
    register_mock_tools(server, state)

    if transport == 'sse':
        app = server.sse_app()
        methods = ('GET', 'POST')
    elif transport == 'streamable-http':
        app = server.streamable_http_app()
        methods = ('POST',)
    else:
        raise ValueError(transport)

    if http_fail:
        app = FlakyHTTPMiddleware(
            app, state, fail_status=fail_status, methods=methods
        )
    return app, server, state


class HttpServerThread:
    def __init__(self, app, port=0):
        self._config = uvicorn.Config(
            app, host='127.0.0.1', port=port, log_level='error'
        )
        self.server = uvicorn.Server(self._config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self):
        self.thread.start()
        for _ in range(500):
            if self.server.started:
                return self
            time.sleep(0.01)
        raise RuntimeError('mock http server failed to start')

    @property
    def port(self):
        return self.server.servers[0].sockets[0].getsockname()[1]

    def stop(self):
        self.server.should_exit = True
        self.thread.join(timeout=5)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc_info):
        self.stop()
