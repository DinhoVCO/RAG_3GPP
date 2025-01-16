from multiprocessing import cpu_count
num_workers = min(2, cpu_count())
print(num_workers)