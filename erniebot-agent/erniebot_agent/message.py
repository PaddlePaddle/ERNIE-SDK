class Message:
    def __init__(self):
        self.content = ''
    
    @property
    def type(self) -> str:
        raise NotImplementedError


class HumanMessage(Message):
    """A Message from a human."""
    
    def __init__(self, content):
        super().__init__()
        self.content = content

    @property
    def type(self) -> str:
        return "human"


class AIMessage(Message):
    """A Message from an AI."""
    
    def __init__(self, content):
        super().__init__()
        self.content = content

    @property
    def type(self) -> str:
        return "ai"