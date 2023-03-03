from Net import Net
import random

net = Net([1, 2, 1])

for i in range(1000):
    a = random.randint(0, 10) / 10
    net.FeedForward([a])
    net.BackProp([a / 2])
    result = net.getResult()
    print(result, "target: ", a / 2)
