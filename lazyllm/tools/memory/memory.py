from .supplier import MemUMemory, Mem0Memory, PowerMemMemory, LocalMemory
from .base import LazyLLMMemoryBase
from lazyllm import globals
from typing import Optional
import lazyllm


class Memory():
    SUPPLIERS = {'memu': MemUMemory,
                 'mem0': Mem0Memory,
                 'powermem': PowerMemMemory,
                 'local': LocalMemory}
    _EXTERNAL_API_KEYS = ('memu_api_key', 'mem0_api_key')

    def __new__(cls, source: Optional[str] = None, *args, **kwargs) -> LazyLLMMemoryBase:
        if source is None:
            for key in cls._EXTERNAL_API_KEYS:
                if lazyllm.config[key]:
                    source = key.rsplit('_api_key', 1)[0]
                    break
            else:
                source = 'local'
        return cls.SUPPLIERS[source](*args, **kwargs)


def memory_hook(query, *inputs, **kw):
    m = Memory()
    output = yield
    m.add(query, output, user_id=globals.get('user_id'), agent_id=globals.get('agent_id'))
