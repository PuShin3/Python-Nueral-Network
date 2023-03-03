from __future__ import annotations
from typing import List
import random
import math

eta = 0.2
alpha = 0.1


class Neuron:
    def __init__(self, netSize: int, id: int) -> None: ...
    def feedForward(self, prevLayer: Layer) -> None: ...
    def calculateOutGradient(self, targetValue: float) -> None: ...
    def calculateHiddenGradient(self, nextLayer: Layer) -> None: ...
    def updateWeight(self, nextLayer: Layer) -> None: ...


# 一列
Layer = List[Neuron]


# 啟動函數
def activationFunction(x: float) -> float:
    return math.tanh(x)


# 斜率
def derivative(x: float) -> float:
    return 1 - x*x


# Neuron跟Neuron中間連接的weight
class Connection:
    def __init__(self, weight: float = 1.0):
        self.weight = weight
        self.deltaWeight = 0


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

        self.gradient = 0

        for i in range(netSize):
            self.weight.append(Connection(random.uniform(0.0, 1.0)))

    def feedForward(self, prevLayer: Layer) -> None:
        """計算outValue

        參數:
            prevLayer (Layer): 前面一列
        """
        self.outValue = 0

        for i in range(len(prevLayer)):
            # value * weight
            self.outValue += prevLayer[i].outValue * \
                prevLayer[i].weight[self.id].weight

        self.outValue = activationFunction(self.outValue)

    def calculateOutGradient(self, targetValue: float) -> None:
        """計算輸出層的gradient

        參數:
            targetValue (float): 實際值
        """

        # 實際值-輸出值
        delta: float = targetValue - self.outValue

        # gradient = delta * derivative
        self.gradient = delta * derivative(self.outValue)

    def calculateHiddenGradient(self, nextLayer: Layer) -> None:
        """計算隱藏層的gradient

        參數:
            nextLayer (Layer): 下一列的Neuron
        """

        # gradient * weight的合
        Sum = 0
        for i in range(len(nextLayer)):
            Sum += nextLayer[i].gradient * self.weight[i].weight

        # gradient = sum * derivative
        self.gradient = Sum * derivative(self.outValue)

    def updateWeight(self, nextLayer: Layer) -> None:
        """更新對於下一列每一個Neuron的權重(weight)

        Args:
            nextLayer (Layer): 下一列的Neuron
        """
        for i in range(len(nextLayer)):
            oldDeltaWeight = self.weight[i].deltaWeight

            newDeltaWeight = (
                eta * nextLayer[i].outValue * nextLayer[i].gradient +  # 更新方向
                alpha * oldDeltaWeight  # 舊的方向(慣性)
            )

            # 更新delta weight
            self.weight[i].deltaWeight = newDeltaWeight

            # 更新weight
            self.weight[i].weight += newDeltaWeight
