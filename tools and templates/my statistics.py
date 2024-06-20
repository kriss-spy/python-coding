li = input().split()
# li = [int(string) for string in li]
li = list(map(lambda x: int(x), li))
average = sum(li) / len(li)
print("average = {}".format(average))
new_li = [(num - average) * (num - average) for num in li]
print(new_li)
variance = sum(new_li) / len(li)
print("variance = {}".format(variance))
