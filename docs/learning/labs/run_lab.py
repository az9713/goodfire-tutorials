"""Small, deterministic teaching experiments; no downloads or API calls.

Run from the archive root: python learning/labs/run_lab.py --lab all
Results are measurements of these fixtures, not replications of Goodfire models.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import time

import numpy as np
import torch
from torch import nn
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
FIGURES = ROOT.parent / 'figures'
SEED = 1729
torch.set_num_threads(2)
plt.rcParams.update({'figure.facecolor': '#101820', 'axes.facecolor': '#16222d',
                     'text.color': '#e9f1f8', 'axes.labelcolor': '#e9f1f8',
                     'xtick.color': '#c7d7e5', 'ytick.color': '#c7d7e5',
                     'axes.edgecolor': '#7991a4', 'font.size': 11,
                     'svg.hashsalt': 'goodfire-study'})


def close(a, b, atol=1e-8):
    np.testing.assert_allclose(a, b, atol=atol, rtol=1e-6)


def plot(name, draw):
    fig, ax = plt.subplots(figsize=(7.5, 4.4), layout='constrained')
    draw(ax)
    fig.savefig(FIGURES / (name + '.svg'), metadata={'Date': None})
    plt.close(fig)


def relu():
    x = torch.tensor([-3., -1., 1., 3.], requires_grad=True)
    hp, hm = torch.relu(x), torch.relu(-x)
    y = hp + hm
    y.sum().backward()
    close(y.detach(), np.abs(x.detach().numpy()))
    close(x.grad, [-1, -1, 1, 1])
    xx = np.linspace(-4, 4, 161)
    def draw(ax):
        ax.plot(xx, np.abs(xx), label='h+ + h−')
        ax.plot(xx, np.maximum(xx, 0), '--', label='ablate h−')
        ax.set(xlabel='x', ylabel='output', title='Constructed absolute-value network')
        ax.legend()
    plot('relu', draw)
    return {'x': x.detach().tolist(), 'output': y.detach().tolist(),
            'gradient': x.grad.tolist(), 'minus3_ablate_negative_branch': 0,
            'claim': 'Exact constructed network; no trained-model result.'}


def coordinates():
    theta = np.pi / 4
    q = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    x = np.array([[1., 2.], [-2., 1.]])
    w = np.array([[2., -1.]])
    close(x @ w.T, (x @ q.T) @ (w @ q.T).T)
    zt = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    h = zt * np.array([10., 1.])
    cov = h.T @ h / len(h)
    close(cov, np.diag([100, 1]))
    close((h[:, 0] * h[:, 1]).mean(), 0)
    def draw(ax):
        ax.scatter(h[:, 0], h[:, 1], c=zt[:, 1], cmap='coolwarm', s=80)
        ax.set(xlabel='10z: dominant variance', ylabel='t: task label',
               title='PCA can discard the task-bearing coordinate', ylim=(-2, 2))
    plot('coordinates', draw)
    return {'covariance': cov.tolist(), 'top_pc_variance_fraction': 100 / 101,
            'best_label_accuracy_from_first_coordinate': .5,
            'label_accuracy_from_second_coordinate': 1.0,
            'linear_interface_rotation_verified': True}


def interventions():
    def restoration(clean, corrupt, patched):
        if clean == corrupt:
            raise ValueError('Restoration is undefined for zero denominator')
        return (patched - corrupt) / (clean - corrupt)
    close(restoration(.8, .2, .5), .5)
    close(restoration(.6, .5, .8), 3)
    try:
        restoration(1, 1, 1)
        raise AssertionError('Expected a zero-denominator error')
    except ValueError:
        pass
    table = [{'ablate_first': a, 'ablate_second': b,
              'output': max(1 - a, 1 - b)} for a in [0, 1] for b in [0, 1]]
    close([r['output'] for r in table], [1, 1, 1, 0])
    return {'restoration_examples': [.5, 3.0], 'redundancy_ablation': table,
            'zero_denominator_rejected': True}


def statistics():
    precision = lambda p, r, f: p * r / (p * r + (1 - p) * f)
    p1, p2 = precision(.01, .9, .05), precision(.001, .9, .05)
    close(p1, 2 / 13)
    hidden = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    output = hidden[:, 0]
    labels = hidden[:, 1] > 0
    probe_prediction = hidden @ np.array([0., 1.]) > 0
    intervened = hidden.copy()
    intervened[:, 1] *= -1
    close(output, intervened[:, 0])
    return {'precision_prevalence_1pct': p1,
            'precision_prevalence_0_1pct': p2,
            'zero_failures_n1000_one_sided_95pct_upper': 1 - .05 ** .001,
            'constructed_unused_coordinate_probe_accuracy': float((probe_prediction == labels).mean()),
            'output_change_after_flipping_probed_coordinate': float(np.max(np.abs(output - intervened[:, 0]))),
            'assumption': 'Independent identically distributed Bernoulli trials.'}


def geometry():
    angles = np.arange(128) * 2 * np.pi / 128
    circle = np.c_[np.cos(angles), np.sin(angles)]
    chord_mid = (np.array([1., 0.]) + np.array([0., 1.])) / 2
    close(np.linalg.norm(chord_mid), 1 / np.sqrt(2))
    a, b, modulus = 2, 3, 12
    u = lambda k: np.array([np.cos(2 * np.pi * k / modulus),
                            np.sin(2 * np.pi * k / modulus)])
    th = 2 * np.pi * b / modulus
    rotation = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    close(rotation @ u(a), u(a + b))
    def draw(ax):
        ax.plot(circle[:, 0], circle[:, 1], label='unit circle')
        ax.plot([1, 0], [0, 1], '--', label='linear interpolation')
        ax.scatter([.5], [.5], label='chord midpoint')
        ax.set(xlabel='coordinate 1', ylabel='coordinate 2', aspect='equal',
               title='A chord leaves the activation manifold')
        ax.legend()
    plot('geometry', draw)
    return {'circle_matrix_rank': int(np.linalg.matrix_rank(circle)),
            'chord_midpoint_radius': float(np.linalg.norm(chord_mid)),
            'mod12_2_plus3': (rotation @ u(a)).tolist()}


def attention():
    q = np.array([1., 0.])
    keys = np.array([[2., 0.], [0., 2.]])
    logits = keys @ q / np.sqrt(2)
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    values = np.array([3., 1.])
    original, swapped = weights @ values, weights @ values[::-1]
    close(original + swapped, 4)
    close(np.array([.75, .25]) @ np.array([4., 0.]), 3)
    return {'weights': weights.tolist(), 'original_values_output': float(original),
            'swapped_values_output_same_attention': float(swapped),
            'interpretation': 'Attention weights alone do not specify the output.'}


def sparse():
    rng = np.random.default_rng(SEED)
    d = rng.normal(size=(5, 9))
    d /= np.linalg.norm(d, axis=0)
    z0 = np.array([1., 0, 0, -.7, 0, 0, 0, 0, 0])
    x = d @ z0
    lam = .03
    step = 1 / np.linalg.norm(d, 2) ** 2
    z = np.zeros(9)
    objective = lambda zz: .5 * np.sum((d @ zz - x) ** 2) + lam * np.abs(zz).sum()
    history = [objective(z)]
    for _ in range(1000):
        v = z - step * d.T @ (d @ z - x)
        z = np.sign(v) * np.maximum(np.abs(v) - step * lam, 0)
        history.append(objective(z))
    assert history[-1] < history[0]
    assert np.max(np.diff(history)) < 1e-10
    close(d @ z, (10 * d) @ (z / 10))
    plot('sparse', lambda ax: (ax.plot(history), ax.set(
        xlabel='ISTA update', ylabel='objective', title='Sparse coding on a fixed dictionary')))
    return {'initial_objective': history[0], 'final_objective': history[-1],
            'relative_reconstruction_error': float(np.linalg.norm(d @ z - x) / np.linalg.norm(x)),
            'coefficients': z.tolist(), 'scale10_penalty_ratio': .1,
            'scope': 'Fixed dictionary inference, not trained SAE or causal recovery.'}


def parameters():
    w = np.array([[2., 1.], [1., 3.]])
    c1 = np.array([[2., 0.], [0., 0.]])
    c2 = w - c1
    close(c1 + c2, w)
    close(100 * w - 99 * w, w)
    e = np.diag([0., .01])
    narrow = np.array([1., 0.])
    test = np.array([0., 1.])
    close(e @ narrow, [0, 0])
    return {'matrix': w.tolist(), 'component1': c1.tolist(), 'component2': c2.tolist(),
            'cancellation_components_relative_norms': [100, 99],
            'residual_error_on_narrow_test': float(np.linalg.norm(e @ narrow)),
            'residual_error_on_orthogonal_input': float(np.linalg.norm(e @ test)),
            'scope': 'Identifiability and coverage fixtures, not SPD/VPD replication.'}


def data():
    differences = np.array([[1., 1.], [1., 1.], [1., -1.]])
    theta = np.zeros(2)
    def loss(t):
        return np.logaddexp(0, -(differences @ t)).mean()
    analytic = -.5 * differences.mean(axis=0)
    eps = 1e-5
    finite = np.array([(loss(theta + eps * e) - loss(theta - eps * e)) / (2 * eps)
                       for e in np.eye(2)])
    close(analytic, finite)
    return {'preference_difference_mean': differences.mean(axis=0).tolist(),
            'loss_gradient_at_zero': analytic.tolist(), 'finite_difference': finite.tolist(),
            'balanced_spurious_coordinate_gradient': 0.,
            'scope': 'Linear logistic preference surrogate with beta=1; not full DPO training.'}


def monitoring():
    prevalence, recall, fpr, judge_recall = .1, .9, .1, .95
    routed = prevalence * recall + (1 - prevalence) * fpr
    expected_cost = .01 + routed
    close(routed, .18)
    return {'routed_fraction': routed, 'expected_cost_relative_to_judge_all': expected_cost,
            'idealized_speedup': 1 / expected_cost,
            'end_to_end_recall_given_conditional_judge_recall': recall * judge_recall,
            'scope': 'Analytic cascade example; conditional judge recall is an assumption.'}


def biology():
    a = np.array([[-1., -1.], [1., 1.]])
    b = np.array([[-1., 1.], [1., -1.]])
    close(a.mean(axis=0), b.mean(axis=0))
    ma, mb = a.T @ a / 2, b.T @ b / 2
    assert not np.allclose(ma, mb)
    return {'mean_A': a.mean(axis=0).tolist(), 'mean_B': b.mean(axis=0).tolist(),
            'second_moment_A': ma.tolist(), 'second_moment_B': mb.tolist(),
            'scope': 'Two synthetic embedding sequences; no genomic or clinical claim.'}


def vision():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    def samples(n):
        y = np.arange(n) % 2
        x = rng.normal(0, .15, (n, 1, 12, 12)).astype('float32')
        for i, label in enumerate(y):
            location = int(rng.integers(2, 9))
            if label == 0:
                x[i, 0, :, location:location + 2] += 1
            else:
                x[i, 0, location:location + 2, :] += 1
        return torch.tensor(x), torch.tensor(y)
    x, y = samples(480)
    xt, yt = samples(160)
    conv, head = nn.Conv2d(1, 4, 3, padding=1), nn.Linear(4, 2)
    model = nn.Sequential(conv, nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), head)
    optimizer = torch.optim.Adam(model.parameters(), lr=.03)
    losses = []
    for _ in range(160):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    with torch.no_grad():
        h = torch.relu(conv(xt)).mean(dim=(2, 3))
        accuracy = lambda hh: (head(hh).argmax(1) == yt).float().mean().item()
        baseline = accuracy(h)
        ablated = []
        for channel in range(4):
            masked = h.clone()
            masked[:, channel] = 0
            ablated.append(accuracy(masked))
    plot('vision', lambda ax: (ax.plot(losses), ax.set(
        xlabel='full-batch training step', ylabel='cross entropy', title='Generated stripe classification')))
    torch.save(model.state_dict(), RESULTS / 'vision-weights.pt')
    return {'train_examples': 480, 'test_examples': 160, 'steps': 160,
            'initial_loss': losses[0], 'final_loss': losses[-1],
            'test_accuracy': baseline, 'test_accuracy_by_ablated_channel': ablated,
            'scope': 'Generated IID stripe data; small CNN; no natural-image generalization claim.'}


def routing():
    torch.manual_seed(SEED)
    inputs = torch.tensor([[a, b, s] for a in range(8) for b in range(8) for s in [8, 9]])
    targets = torch.tensor([a if s == 8 else b for a in range(8) for b in range(8) for s in [8, 9]])
    order = torch.randperm(128)
    train, test = order[:104], order[104:]
    class Router(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(10, 16)
            self.position = nn.Parameter(torch.randn(3, 16) * .02)
            layer = nn.TransformerEncoderLayer(16, 2, 32, dropout=0, batch_first=True)
            self.transformer = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
            self.head = nn.Linear(16, 8)

        def forward(self, tokens, erase_position=None):
            h = self.embed(tokens) + self.position
            if erase_position is not None:
                h = h.clone()
                h[:, erase_position] = 0
            return self.head(self.transformer(h)[:, 2])
    model = Router()
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    losses = []
    for _ in range(400):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(inputs[train]), targets[train])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    model.eval()
    with torch.no_grad():
        accuracy = lambda idx, erase=None: (model(inputs[idx], erase).argmax(1) == targets[idx]).float().mean().item()
        training, testing = accuracy(train), accuracy(test)
        erased = [accuracy(test, position) for position in range(3)]
    plot('routing', lambda ax: (ax.plot(losses), ax.set(
        xlabel='full-batch training step', ylabel='cross entropy', title='Tiny transformer token routing')))
    torch.save(model.state_dict(), RESULTS / 'routing-weights.pt')
    return {'train_examples': 104, 'test_examples': 24, 'steps': 400,
            'train_accuracy': training, 'test_accuracy': testing,
            'test_accuracy_erasing_each_input_position': erased,
            'final_loss': losses[-1],
            'scope': 'One split and seed; erasure can be off-distribution; not natural-language circuit evidence.'}


LABS = {f.__name__: f for f in [relu, coordinates, interventions, statistics,
        geometry, attention, sparse, parameters, data, monitoring, biology, vision, routing]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lab', choices=['all'] + list(LABS), default='all')
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    selected = LABS if args.lab == 'all' else {args.lab: LABS[args.lab]}
    for name, function in selected.items():
        start = time.perf_counter()
        result = function()
        receipt = {'lab': name, 'seed': SEED, 'device': 'cpu',
                   'python': platform.python_version(), 'numpy': np.__version__,
                   'torch': torch.__version__, 'elapsed_seconds': time.perf_counter() - start,
                   'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'checks': 'passed', 'result': result}
        (RESULTS / (name + '.json')).write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
        print(name + ': ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
