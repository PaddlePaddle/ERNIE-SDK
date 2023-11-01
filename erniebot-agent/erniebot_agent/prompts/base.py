from abc import ABC, abstractmethod


class BasePromptTemplate(ABC):
    
    def __init__(self, input_variabels):
        self.input_variabels: list[str] = input_variabels
    
    @abstractmethod
    def format(self, **kwargs):
        raise NotImplementedError
    
    @abstractmethod    
    def format_prompt(self,):
        raise NotImplementedError