from __future__ import annotations
from typing import List


class Neuron:
    def __init__(self, netSize: int, id: int) -> None: ...
    def feedForward(self, prevLayer: Layer) -> None: ...


# 一列
Layer = List[Neuron]


# Neuron跟Neuron中間連接的weight
class Connection:
    def __init__(self, weight: float = 1.0):
        self.weight = weight


class Neuron:
    def __init__(self, netSize: int, id: int):
        """創建一個Neuron

        參數:
            netSize (int): 下一列的行數
            id (int): 在這列所屬的第幾個
        """

        # 輸出值
        self.outValue: float = 0

        self.id = id

        # 創造weight的陣列
        self.weight: List[Connection] = []
        for i in range(netSize):
            self.weight.append(Connection())

    def feedForward(self, prevLayer: Layer):
        """計算outValue

        參數:
            prevLayer (Layer): 前面一列
        """

        for i in range(len(prevLayer)):
            # value * weight
            self.outValue += prevLayer[i].outValue * \
                prevLayer[i].weight[self.id].weight
