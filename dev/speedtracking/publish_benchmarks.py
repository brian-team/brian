from datetime import date
from matplotlib import pyplot as plt
from run_benchmarks import benchmarks, DB_PATH

#nothing fancy for now:
for idx, benchmark in enumerate(benchmarks):
    benchmark.plot(DB_PATH)
    plt.savefig('./benchmark_%d_%s.png' % (idx, date.today().isoformat()),
                dpi=150)
    