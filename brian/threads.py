# ----------------------------------------------------------------------------------
# Copyright ENS, INRIA, CNRS
# Contributors: Romain Brette (brette@di.ens.fr) and Dan Goodman (goodman@di.ens.fr)
# 
# Brian is a computer program whose purpose is to simulate models
# of biological neural networks.
# 
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software.  You can  use, 
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info". 
# 
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability. 
# 
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or 
# data to be ensured and,  more generally, to use and operate it in the 
# same conditions as regards security. 
# 
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# ----------------------------------------------------------------------------------
# 
'''
EXPERIMENTAL

Threading system.
Group updating is done in parallel threads.
Goal: speed-up with dual core machines.
For the moment the threads seem to run on the same processor
(I don't know how to change this behaviour). 
'''
from multiprocessing import Process,Queue

__all__=['GroupUpdater']
   
class GroupUpdater(Process):
    '''
    An object that updates groups.
    Groups are fed into a queue by the main thread.
    Each group updater is run into is own thread and
    gets groups to update from the queue.
    '''
    def __init__(self,queue):
        Process.__init__(self)
        self.queue=queue   
          
    def run(self):
        P=0
        while P!=None:
            P=self.queue.get()
            #print id(self),"is updating",id(P)
            if P!=None:
                P.update()
                P.reset()
                self.queue.task_done()