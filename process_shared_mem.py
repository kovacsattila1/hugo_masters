import multiprocessing
manager = multiprocessing.Manager()
final_list = manager.list()

input_list_one = ['one', 'two', 'three', 'four', 'five']
input_list_two = ['six', 'seven', 'eight', 'nine', 'ten']

def worker(data):
    for item in data:
        final_list.append(item)

process1 = multiprocessing.Process(target=worker, args=[input_list_one])
process2 = multiprocessing.Process(target=worker, args=[input_list_two])

process1.start()
process2.start()
process1.join()
process2.join()

print(final_list)