from cluster_modelfitting_fast import *

if __name__=='__main__':
    cluster_worker_script(light_worker,
                          named_pipe=True)
    