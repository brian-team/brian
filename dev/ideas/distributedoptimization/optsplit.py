from numpy import *

__all__ = ['OptSplit']

class OptSplit:
    def __init__(self, worker_size, group_size, verbose = False):
        """Internal class used to split a multiprocessing optimization of several groups
        in parallel across different workers with minimum data transfer.
        
        Initialized with arguments:
        
        ``worker_size``
            A list containing the number of particles running over each worker.
        ``group_size``
            A list containing the number of particles in each group.
            
        **Methods**
        
        .. method:: split_groups()
        
            Creates a property self.groups_by_worker containing the list of groups
            on each worker with the number of particles inside them.
        
        .. method:: combine_items(items)
        
            Finds the best items within each group from their fitness values
            splitted among the workers.
        
        .. method:: split_items(items)
        
            Sends the best items to the workers from the best items computed with
            :func:`combine_items`.
        """
        if sum(array(worker_size)) <> sum(array(group_size)):
            raise Exception('The total number of particles should be the same in worker_size and group_size')
        # Number of particles by worker (list)
        self.worker_size = worker_size
        # Total number of workers
        self.worker_number = len(worker_size)
        # Size of groups
        self.group_size = group_size
        # Total number of groups of particles
        self.group_number = len(group_size)
        # Total number of particles
        self.particles = sum(worker_size)
        self.split_groups()
        bworkers = ''
        if len(worker_size)>1:
            bworkers = 's'
        bgroups = ''
        if len(group_size)>1:
            bgroups = 's'
        if verbose:
            print "%d worker%s with size%s" % (len(worker_size), bworkers, bworkers), worker_size
            print "%d group%s with size%s" % (len(group_size), bgroups, bgroups), group_size
            print
        
    def split_groups(self):
        """
        Returns the repartition of groups over workers.
        The result is a vector. result[j] contains the repartition
        of the particles within worker j into subgroups : it is
        a dictionary (i, n)
        where i is the group index and n the number of particles in
        the subgroup.
        Same thing for workers_by_group.
        We have: 
            groups_by_worker[worker][group] = n
            workers_by_group[group][worker] = n
        """
        self.groups_by_worker = [dict() for i in range(self.worker_number)]
        self.workers_by_group = [dict() for i in range(self.group_number)]
        
        def fun(n, group, group_size, worker, worker_size):
            if n > min(group_size, worker_size):
                if n > max(group_size, worker_size):
                    n1, n2 = max(group_size, worker_size), n-max(group_size, worker_size)
                else:
                    n1, n2 = min(group_size, worker_size), n-min(group_size, worker_size)
                group, group_size, worker, worker_size = fun(n1, group, group_size, worker, worker_size)
                group, group_size, worker, worker_size = fun(n2, group, group_size, worker, worker_size)
            else:
                self.groups_by_worker[worker][group] = n
                self.workers_by_group[group][worker] = n
                if n == group_size:
                    group += 1
                    if group < len(self.group_size):
                        group_size = self.group_size[group]
                else:
                    group_size -= n
                if n == worker_size:
                    worker += 1
                    if worker < len(self.worker_size):
                        worker_size = self.worker_size[worker]
                else:
                    worker_size -= n
            return group, group_size, worker, worker_size
        
        fun(self.particles, 0, self.group_size[0], 0, self.worker_size[0])

if __name__ == '__main__':
    cs = OptSplit([10, 10], [7, 7, 6])
    print "groups by worker", cs.groups_by_worker
    print "workers by group", cs.workers_by_group
