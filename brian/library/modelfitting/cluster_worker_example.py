from cluster_modelfitting import *

if __name__=='__main__':
    cluster_worker_script(modelfitting_worker,
                          named_pipe=True)
    