iterms = ['A', 'B', 'C', 'D', 'E', 'F']

for item in iterms:
    if item == 'B':
        iterms.remove('B')
    else:
        print(item)