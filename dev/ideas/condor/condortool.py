'''
A tool to submit jobs to Condor
'''
import os

condor_path = "C:\\condor\\bin\\"

os.system(condor_path + " condor_submit brianjob.sub")
