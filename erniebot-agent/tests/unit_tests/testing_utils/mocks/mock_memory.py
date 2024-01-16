from erniebot_agent.memory.messages import SystemMessage


class FakeMemory(object):
    def __init__(self):
        super().__init__()
        self._history = []

    def add_messages(self, messages):
        for message in messages:
            self.add_message(message)

    def add_message(self, message):
        self._history.append(message)

    def get_messages(self):
        return self._history[:]

    def clear_chat_history(self):
        self._history.clear()

    def set_system_message(self, message: SystemMessage):
        pass
