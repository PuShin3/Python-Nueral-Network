from Net import Net

net = Net([2, 2, 1])
net.FeedForward([0.5, 1.0])
result = net.getResult()
print(result)
