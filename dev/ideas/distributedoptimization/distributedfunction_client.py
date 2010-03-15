from distributedfunction import *

def distributedworker(max_cpu = None, max_gpu = None, port = None,
                      named_pipe = None, authkey = 'distributedfunction'):
    cluster_worker_script(DistributedWorker,
                          max_gpu=max_gpu, max_cpu=max_cpu, port=port,
                          named_pipe=named_pipe, authkey=authkey)

if __name__ == '__main__':
    distributedworker(named_pipe=True)
    