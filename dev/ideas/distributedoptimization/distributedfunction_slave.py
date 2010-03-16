from distributedfunction_master import *

if __name__ == '__main__':
    import distributedfunction as df
    df.distributedslave(named_pipe=True)
    