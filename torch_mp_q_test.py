# import torch


# def _process(queue):
#     input_ = queue.get()
#     print('get')
#     queue.put(input_)
#     print('put')


# if __name__ == '__main__':
#     torch.multiprocessing.set_start_method('spawn')
#     input_ = torch.ones(1).cuda()
#     queue = torch.multiprocessing.Queue()
#     queue.put(input_)
#     process = torch.multiprocessing.Process(target=_process, args=(queue,))
#     process.start()
#     process.join()
#     result = queue.get()
#     print('end')
#     print(result)


import multiprocessing as mp
import time

def foo(q):
    print("foo started")
    msg = q.get()
    print(msg)
    q.put('hello')

if __name__ == '__main__':
    mp.set_start_method('spawn')
    q = mp.Queue()
    p = mp.Process(target=foo, args=(q,))
    p.start()
    time.sleep(5)
    q.put("hello from main")
    print(q.get())
    p.join()