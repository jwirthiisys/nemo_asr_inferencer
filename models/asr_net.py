from abc import ABC, abstractmethod
class AsrNet(ABC):
    def __init__(self, device) -> None:
        self.device = device
    
    @abstractmethod
    def batch_inference(self, inputs):
        pass
    
    @abstractmethod
    def inference(self, input) -> str:
        pass
    
    @abstractmethod
    def train(self, train_set, validation_set, train_config):
        pass