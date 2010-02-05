'''
Runs all of the examples in the examples/ folder.

This version clears any history of which examples did or didn't pass.
Better to use allexamples_cumulative but delete the examples_completed.txt
file first.
'''

open('examples_completed.txt', 'w').write('')
import allexamples_cumulative
