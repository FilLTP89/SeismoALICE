import core.app
from configuration.session import setup

class Seismo(metaclass=core.app.SingletonMeta):
    def set_setup(self):
        self._setup = session.setup()
    
    def get_setup(self):
        return self._setup