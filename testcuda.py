import torch
import timeit
import torch.utils.benchmark as benchmark

# ref : https://pytorch.org/tutorials/recipes/recipes/benchmark.html

print('# Define and check benchmarking fuction')


def batched_dot_mul_sum(a, b):
    '''Computes batched dot by multiplying and summing'''
    return a.mul(b).sum(-1)


def batched_dot_bmm(a, b):
    '''Computes batched dot by reducing to bmm'''
    a = a.reshape(-1, 1, a.shape[-1])
    b = b.reshape(-1, b.shape[-1], 1)
    return torch.bmm(a, b).flatten(-3)


# Input for benchmarking
x = torch.randn(100, 64)
# Ensure that both functions compute the same output
assert batched_dot_mul_sum(x, x).allclose(batched_dot_bmm(x, x))

cuda_avail = torch.cuda.is_available()

if (cuda_avail):
    print(
        f'# CUDA is available, device is {torch.cuda.get_device_name()} ({torch.cuda.current_device()})')

num_threads = 1
num_threads = torch.get_num_threads()


def timeit_run(x):
    # no warmup and n threads support
    t0 = timeit.Timer(
        stmt='batched_dot_mul_sum(x, x)',
        setup='from __main__ import batched_dot_mul_sum',
        globals={'x': x})

    t1 = timeit.Timer(
        stmt='batched_dot_bmm(x, x)',
        setup='from __main__ import batched_dot_bmm',
        globals={'x': x})

    print(f'>> timeit mul_sum(x, x): {t0.timeit(100) / 100 * 1e6:>5.1f} us')
    print(f'>> timeit bmm(x, x):     {t1.timeit(100) / 100 * 1e6:>5.1f} us')


def torch_benchmark_info(m: benchmark.Measurement):
    time_unit, time_scale = benchmark.select_unit(m.median)
    n = len(m._sorted_times)
    str = f"""{m.median / time_scale:.2f} {time_unit} ({n} measurement{'s' if n > 1 else ''}, {m.number_per_run} runs {'per measurement,' if n > 1 else ','} {m.num_threads} thread{'s' if m.num_threads > 1 else ''})"""
    return str


def torch_benchmark_run(x):
    t0 = benchmark.Timer(
        stmt='batched_dot_mul_sum(x, x)',
        setup='from __main__ import batched_dot_mul_sum',
        globals={'x': x},
        num_threads=num_threads)

    t1 = benchmark.Timer(
        stmt='batched_dot_bmm(x, x)',
        setup='from __main__ import batched_dot_bmm',
        globals={'x': x},
        num_threads=num_threads)

    m = t0.timeit(100)
    print(f'>> torch_benchmark mul_sum(x, x): {torch_benchmark_info(m)})')

    m = t1.timeit(100)
    print(f'>> torch_benchmark bmm(x, x):     {torch_benchmark_info(m)})')


print(f'# Benchmarking on cpu')
x = torch.randn(10000, 1024, device='cpu')
# timeit_run(x)
torch_benchmark_run(x)

if (cuda_avail):
    print(f'# Benchmarking on cuda')
    x = torch.randn(10000, 1024, device='cuda')
    # timeit_run(x)
    torch_benchmark_run(x)
