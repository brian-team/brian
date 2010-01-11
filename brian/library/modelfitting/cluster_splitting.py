from numpy import *

class ClusterSplitting:
    def __init__(self, worker_size, group_size):
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
        print "%d workers with sizes" % len(worker_size), worker_size
        print "%d groups with sizes" % len(group_size), group_size
        
    def split_groups(self):
        """
        Returns the repartition of groups over workers.
        The result is a vector. result[j] contains the repartition
        of the particles within worker j into subgroups : it is
        a list of pairs (i, n)
        where i is the group index and n the number of particles in
        the subgroup.
        """
        self.groups_by_worker = [[] for i in range(self.worker_number)]
        
        def fun(n, group, group_size, worker, worker_size):
            if n > min(group_size, worker_size):
                if n > max(group_size, worker_size):
                    n1, n2 = max(group_size, worker_size), n-max(group_size, worker_size)
                else:
                    n1, n2 = min(group_size, worker_size), n-min(group_size, worker_size)
                group, group_size, worker, worker_size = fun(n1, group, group_size, worker, worker_size)
                group, group_size, worker, worker_size = fun(n2, group, group_size, worker, worker_size)
            else:
                self.groups_by_worker[worker].append((group, n))
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

    def combine_items(self, items):
        """
        Returns the best items from their fitness over the workers.
        items[i] is a list of triplets (group, item, value) where value is the
        fitness of item in the given group, for worker i.
        best_items is a list of triplets (worker, best_item, best_value)
        """
        all_items = []
        [all_items.extend(tuples) for tuples in items]
        
        best_items = []    
        for i in range(self.group_number):
            items, values = zip(*[(item, value) for (j, item, value) in all_items if j==i])
            items = array(items)
            values = array(values)
            best = nonzero(values == max(values))[0][0]
            item, value = items[best], values[best]
            best_items.append((i, item, value))
        return best_items

    def split_items(self, items):
        """
        Splits items over workers.
        items is a list of triplets (group, item, value)
        splitted_items is the list of best items, 
        splitted_items[worker] is a list of best items for each subgroup within the worker
        """
        splitted_items = []
        for i in range(self.worker_number):
            worker_items = []
            for (group, n) in self.groups_by_worker[i]:
                # Appends item when items[group] = group, item, value
                worker_items.append(items[group][1])
            splitted_items.append(worker_items)
        return splitted_items

#    def expand_items(self, worker, items):
#        """
#        If items are column vectors, expanded_items are matrices
#        tiled n times along the x-axis, where n is the size of the 
#        subgroup.
#        """
#        result = None
#        # subgroups and their capacity in current worker
#        groups, n_list = zip(*self.groups_by_worker[worker])
#        
#        for n, item in zip(n_list, items):
#            tiled = tile(item.reshape((-1,1)),(1,n))
#            if result is None:
#                result = tiled
#            else:
#                result = hstack((result, tiled))
#        return result

if __name__ == '__main__':
    from numpy.random import *
    cs = ClusterSplitting([10, 10], [7, 7, 6])
    items = [[(0,rand(3,1),10),(1,rand(3,1),20)],[(1,rand(3,1),30),(2,rand(3,1),40)]]
    combined_items = cs.combine_items(items)
    splitted_items = cs.split_items(combined_items)
    print items
    print combined_items
    print splitted_items
    print cs.expand_items(0, [item for group, item, value in combined_items])
    