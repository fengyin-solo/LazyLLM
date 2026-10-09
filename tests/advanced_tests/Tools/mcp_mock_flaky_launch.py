import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp_mock_servers import create_server, register_mock_tools  # noqa: E402


def _read_count(path):
    try:
        with open(path, 'r') as handle:
            return int(handle.read().strip() or '0')
    except (FileNotFoundError, ValueError):
        return 0


def _write_count(path, count):
    with open(path, 'w') as handle:
        handle.write(str(count))


count_file = os.environ['MCP_LAUNCH_COUNT_FILE']
fail_n = int(os.environ.get('MCP_LAUNCH_FAIL_N', '1'))
count = _read_count(count_file) + 1
_write_count(count_file, count)

if count <= fail_n:
    sys.stderr.write(f'simulated launch failure #{count}\n')
    sys.exit(1)

state = {}
server = create_server('mcp-mock-stdio')
register_mock_tools(server, state)

if __name__ == '__main__':
    if hasattr(server, 'run_stdio_async'):
        asyncio.run(server.run_stdio_async())
    else:
        server.run(transport='stdio')
