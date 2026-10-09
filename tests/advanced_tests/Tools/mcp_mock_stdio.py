import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp_mock_servers import create_server, register_mock_tools  # noqa: E402

state = {}
server = create_server('mcp-mock-stdio')
register_mock_tools(server, state)

if __name__ == '__main__':
    if hasattr(server, 'run_stdio_async'):
        asyncio.run(server.run_stdio_async())
    else:
        server.run(transport='stdio')
