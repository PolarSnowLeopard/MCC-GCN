"""Check that pre-computed NPZ padding width does not affect PyG conversion.

The loader should use each sample's graph_size and ignore padded nodes. This
script repads a few samples to different widths, then compares converted PyG
graphs and, optionally, model logits.
"""
import argparse
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mcc_gcn.data.dataset import GraphDataLoader
from mcc_gcn.models.gcn import GCNNet


def parse_args():
    p = argparse.ArgumentParser(description="Validate padding-width invariance")
    p.add_argument('--data', default='data/HKU_data_6_experiment_1.npz')
    p.add_argument('--model', default=None, help='Optional checkpoint for logit comparison')
    p.add_argument('--samples', type=int, default=10)
    p.add_argument('--extra-pad', type=int, default=17)
    p.add_argument('--num-classes', type=int, default=4)
    p.add_argument('--atol', type=float, default=1e-6)
    return p.parse_args()


def _repadded_npz(src, out_path, sample_count, extra_pad):
    data = np.load(src, allow_pickle=True)
    count = min(sample_count, len(data['labels']))
    graph_size = data['graph_size'][:count].astype(np.int32)
    width = int(graph_size.max()) + extra_pad

    V = np.zeros((count, width, data['V'].shape[2]), dtype=data['V'].dtype)
    A = np.zeros((count, width, data['A'].shape[2], width), dtype=data['A'].dtype)
    masks = np.zeros((count, width, 1), dtype=data['masks'].dtype)

    for ix, size in enumerate(graph_size):
        V[ix, :size] = data['V'][ix, :size]
        A[ix, :size, :, :size] = data['A'][ix, :size, :, :size]
        masks[ix, :size] = 1

    save = {
        'V': V,
        'A': A,
        'labels': data['labels'][:count],
        'masks': masks,
        'graph_size': graph_size,
        'tags': data['tags'][:count],
    }
    if 'subgraph_size' in data:
        save['subgraph_size'] = data['subgraph_size'][:count]
    if 'global_state' in data:
        save['global_state'] = data['global_state'][:count]
    np.savez(out_path, **save)


def _assert_graphs_equal(left, right, atol):
    if len(left) != len(right):
        raise AssertionError(f"sample count mismatch: {len(left)} != {len(right)}")

    for ix, (a, b) in enumerate(zip(left, right)):
        if a.x.shape != b.x.shape:
            raise AssertionError(f"sample {ix}: x shape mismatch {a.x.shape} != {b.x.shape}")
        if a.edge_index.shape != b.edge_index.shape:
            raise AssertionError(
                f"sample {ix}: edge_index shape mismatch {a.edge_index.shape} != {b.edge_index.shape}"
            )
        if a.edge_attr.shape != b.edge_attr.shape:
            raise AssertionError(
                f"sample {ix}: edge_attr shape mismatch {a.edge_attr.shape} != {b.edge_attr.shape}"
            )
        if not torch.allclose(a.x, b.x, atol=atol, rtol=0):
            raise AssertionError(f"sample {ix}: x values differ")
        if not torch.equal(a.edge_index, b.edge_index):
            raise AssertionError(f"sample {ix}: edge_index values differ")
        if not torch.allclose(a.edge_attr, b.edge_attr, atol=atol, rtol=0):
            raise AssertionError(f"sample {ix}: edge_attr values differ")


def _load_model(path, num_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCNNet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model, device


def _assert_logits_equal(model, device, left, right, atol):
    with torch.no_grad():
        for ix, (a, b) in enumerate(zip(left, right)):
            batch_a = torch.zeros(a.x.size(0), dtype=torch.long, device=device)
            batch_b = torch.zeros(b.x.size(0), dtype=torch.long, device=device)
            out_a = model(a.x.to(device), a.edge_index.to(device), batch_a)
            out_b = model(b.x.to(device), b.edge_index.to(device), batch_b)
            if not torch.allclose(out_a, out_b, atol=atol, rtol=0):
                diff = float((out_a - out_b).abs().max().cpu())
                raise AssertionError(f"sample {ix}: logits differ, max abs diff={diff}")


def main():
    args = parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        repadded = os.path.join(tmp, 'repadded.npz')
        _repadded_npz(args.data, repadded, args.samples, args.extra_pad)

        original = GraphDataLoader(args.data).pyg_data[:args.samples]
        variant = GraphDataLoader(repadded).pyg_data

        _assert_graphs_equal(original, variant, args.atol)
        print(f"PyG graph invariance OK for {len(variant)} samples")
        print("node counts:", [int(g.x.size(0)) for g in variant])

        if args.model:
            model, device = _load_model(args.model, args.num_classes)
            _assert_logits_equal(model, device, original, variant, args.atol)
            print(f"model logit invariance OK on {device}")


if __name__ == '__main__':
    main()
