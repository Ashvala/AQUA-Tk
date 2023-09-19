from abc import ABC, abstractmethod
import numpy as np


class Extractor(ABC):
    def __init__(self, *args, **kwargs):
        """
        Initialize the extractor
        :param args: Some arguments you can pass in
        :param kwargs: Some keyword arguments you can pass in
        """
        pass

    @abstractmethod
    def get_embeddings(self, x, sr=16000) -> np.ndarray:
        """
        Extract embeddings
        :param x: the audio signal
        :param sr: sampling rate
        :return:
        """
        pass

    @abstractmethod
    def cleanup(self):
        pass
