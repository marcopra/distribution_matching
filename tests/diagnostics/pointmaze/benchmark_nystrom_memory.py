"""Measure the streamed actor at production support sizes, without an environment."""
import argparse
import gc
import json
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
import torch.nn.functional as F

from agent.rover_matchers import DistributionMatcher, NystromActorModel
from agent.rover_sink import kernel_sink_gram


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, nargs="+", default=[60004, 80001])
    parser.add_argument("--landmarks", type=int, default=15000)
    parser.add_argument("--features", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=1000)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=Path("tests/outputs/pointmaze/nystrom_streaming_memory.json"))
    args = parser.parse_args()
    assert torch.cuda.is_available(), "CUDA benchmark requires a GPU"
    torch.manual_seed(1)
    torch.set_num_threads(8)
    results = []
    for n in args.samples:
        assert n >= args.landmarks > 0
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        x = torch.rand(n, args.features, device="cuda", dtype=torch.float64)
        x /= x.sum(1, keepdim=True)
        y, y_next = x[:args.landmarks].clone(), x[-args.landmarks:].clone()
        actions = torch.arange(n, device="cuda") % 4
        E = F.one_hot(actions, 4).to(x.dtype)
        alpha = x.new_zeros(args.landmarks, 1)
        alpha[0] = 1
        coverage = torch.rand(args.landmarks, 1568, device="cuda", dtype=x.dtype)
        coverage /= coverage.sum(1, keepdim=True)
        matcher = DistributionMatcher(.001, gamma=.99, kernel_type="gaussian", kernel_bandwidth=.0547721)
        gram = kernel_sink_gram(matcher.state_kernel, coverage, y.sum(1), .1, args.block_size)
        model = NystromActorModel(matcher, x, y, y_next, actions, actions[:args.landmarks],
                                 E, alpha, gram, args.rank, args.block_size)
        pi = model.policy(x.new_zeros(n + 1, 1))
        loss = model.loss(pi)
        gradient = model.gradient_coefficients(pi)
        diagnostics = model.diagnostics(pi)
        assert torch.isfinite(loss) and torch.isfinite(gradient).all()
        torch.cuda.synchronize()
        row = dict(samples=n, landmarks=args.landmarks, features=args.features, rank=args.rank,
                   dtype=str(x.dtype), seconds=time.perf_counter()-started,
                   allocated_peak=torch.cuda.max_memory_allocated(),
                   reserved_peak=torch.cuda.max_memory_reserved(),
                   rss_peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                   loss=float(loss), diagnostics=diagnostics)
        del model, gram, gradient, loss, pi, coverage, x, y, y_next, E, alpha, actions
        gc.collect()
        torch.cuda.synchronize()
        row["allocated_after_release"] = torch.cuda.memory_allocated()
        results.append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2)+"\n")
        print(json.dumps(row), flush=True)
        assert row["allocated_peak"] < 20 * 1024**3, "Actor exceeded 20 GiB target"


if __name__ == "__main__":
    main()
