from __future__ import annotations

import zstandard

from abc import ABC, abstractmethod
from typing import List

ENCODING = "UTF-8"

class CompressorselfAdd(ABC):

    @abstractmethod
    def fit(self, texts: List[str]) -> CompressorselfAdd:
        raise NotImplementedError()

    @abstractmethod
    def get_compressed_len(self, text: str):
        raise NotImplementedError()

    @abstractmethod
    def dictionary_size(self) -> int:
        raise NotImplementedError()


class ZstdCompressorSelf(CompressorselfAdd):

    def __init__(self, size: int = -1, compression_level=9):
        self.compression_level = compression_level
        self.size = size

    def fit(self, data: List[str]) -> ZstdCompressorSelf:
        if self.size < -1:
            raise ValueError("size must be -1, 0 or an integer")
        if self.size == -1:
            # -1: special value - the whole dataset is set maintained in memory and set as prefix for compression
            combined_texts = '\n'.join(data)
            self.dictionary = zstandard.ZstdCompressionDict(combined_texts.encode(ENCODING),
                                                            dict_type=zstandard.DICT_TYPE_RAWCONTENT)
        else:
            size_limit = int(1e10) if self.size == 0 else self.size
            try:
              self.dictionary = zstandard.train_dictionary(size_limit, [e.encode(ENCODING) for e in data], split_point=1,
                                                         level=self.compression_level)
            except Exception as e:
                if "Src size is incorrect" in str(e):
                    print("WARNING - Could not train dictionary. Not enough data. Using the whole training data as compressor prefix.")
                    combined_texts = '\n'.join(data)
                    self.dictionary = zstandard.ZstdCompressionDict(combined_texts.encode(ENCODING),
                                                                    dict_type=zstandard.DICT_TYPE_RAWCONTENT)
                else:
                    raise e
        self.dictionary.precompute_compress(level=self.compression_level)
        self.compressor = zstandard.ZstdCompressor(dict_data=self.dictionary)

        return self

    def get_compressed_len(self, text: str):
        compressed = self.compressor.compress(text.encode(ENCODING))
        return len(compressed)

    def dictionary_size(self):
        return len(self.dictionary.as_bytes())


def self_adaption(dataset):
    if dataset in {"kirnews", "R52", "20News", "kinnews", "SwahiliNews"}:
        compressor_level = 1
    else:
        compressor_level = 5
    return compressor_level