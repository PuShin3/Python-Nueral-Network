from Neuron import Neuron, Layer
from typing import List

# Class functions without definition


class Net:
    def __init__(self, size: List[int]) -> None: ...
    def FeedForward(self, inputVal: List[float]) -> None: ...
    def getResult(self) -> List[float]: ...
    def BackProp(self, targetValue: List[float]) -> None: ...


class Net:
    def __init__(self, size: List[int]):
        """創建一個神經網路

        參數:
            size (List[int]): 一維陣列，代表神經網路的架構，包含著每一列的行數。
        """

        # 創造空陣列
        self.layers: List[Layer] = []

        # 依照架構創造一個裝著Neuron的二維陣列
        for i in range(len(size)):
            # 如果不屬於最後一列
            if i+1 < len(size):
                numSize = size[i+1]

            # 屬於最後一列
            else:
                numSize = 0

            # Append一個陣列
            self.layers.append([Neuron(numSize, j) for j in range(size[i])])

    def FeedForward(self, inputVal: List[float]):
        """把輸入的Neuron(第一列的Neuron)的outValue設為InputVal，並且開始由第一列呼叫每一行的feedForward

        參數:
            inputVal (List[float]): Input value
        """

        # 把輸入的Neuron(第一列的Neuron)的outValue設為InputVal
        for i in range(len(self.layers[0])):
            self.layers[0][i].outValue = inputVal[i]

        # 由第一列開始呼叫每一行的feedForward
        for i in range(1, len(self.layers)):
            # 前一列
            prevLayer = self.layers[i-1]

            # 第i列的每一行
            for j in range(len(self.layers[i])):
                self.layers[i][j].feedForward(prevLayer)

    def getResult(self) -> List[float]:
        """回傳最後一列每一行Neuron的值

        回傳:
            List[float]: FeedForward的結果
        """

        # 創造空陣列
        ret = []

        # 最後一列的每一行
        # index為-1的時候代表取陣列從後面數來第一個，-2代表從後面數來第二個，以此類推
        for i in range(len(self.layers[-1])):
            ret.append(self.layers[-1][i].outValue)

        return ret

    def BackProp(self, targetValue: List[float]) -> None:
        """back propagation

        Args:
            targetValue (List[float]): 實際輸出值
        """
        # 輸出層(最後一層)
        OutLayer = self.layers[-1]

        # 計算輸出層gradient
        for i in range(len(OutLayer)):
            OutLayer[i].calculateOutGradient(targetValue[i])

        # 計算隱藏層的gradient
        for i in range(len(self.layers) - 2, 0, -1):
            for j in range(len(self.layers[i])):
                self.layers[i][j].calculateHiddenGradient(self.layers[i+1])

        # 更新每個neuron的weight(輸出層除外)
        for i in range(0, len(self.layers) - 1):
            for j in range(len(self.layers[i])):
                self.layers[i][j].updateWeight(self.layers[i+1])
