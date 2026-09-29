"""Small MF workflow demonstration. Run from the repository root."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from Bio import SeqIO
from Bio.PDB import PDBParser
from torch_geometric.data import Data
from torch_geometric.utils import coalesce

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "DeepAMIGO/human/src_model"
sys.path.insert(0, str(MODEL_DIR))
from model_ph2_bp_adptivea import DeepAMIGO  # noqa: E402
from metrics import compute_cafa_metrics  # noqa: E402


def graph_from_pdb(path, length):
    structure = PDBParser(QUIET=True).get_structure("protein", path)
    chain = next(structure[0].get_chains())
    coords = [r["CA"].get_coord() if "CA" in r else [0, 0, 0]
              for r in chain if r.id[0] == " "]
    coords = torch.tensor(np.asarray(coords), dtype=torch.float32)
    if len(coords) != length:
        coords = torch.zeros(length, 3)
        coords[:, 0] = torch.arange(length) * 3.8
    k = min(10, length - 1)
    distances = torch.cdist(coords, coords)
    distances.fill_diagonal_(float("inf"))
    neighbors = distances.topk(k, largest=False).indices
    src = torch.arange(length).repeat_interleave(k)
    edges = torch.stack([src, neighbors.flatten()])
    seq = torch.arange(length - 1)
    sequence_edges = torch.stack([torch.cat([seq, seq + 1]),
                                  torch.cat([seq + 1, seq])])
    edges = coalesce(torch.cat([edges, sequence_edges], dim=1), num_nodes=length)
    d = torch.norm(coords[edges[0]] - coords[edges[1]], dim=1)
    rbf = torch.exp(-0.1 * (d[:, None] - torch.linspace(0, 20, 16)[None, :]) ** 2)
    return edges, rbf


def parent_map(obo_path, terms):
    term_set = set(terms)
    parents = {}
    current = None
    for line in obo_path.read_text().splitlines():
        if line == "[Term]":
            current = None
        elif line.startswith("id: GO:"):
            current = line[4:].strip()
            parents.setdefault(current, [])
        elif current and (line.startswith("is_a: GO:") or
                          line.startswith("relationship: part_of GO:")):
            parent = next((part for part in line.split() if part.startswith("GO:")), None)
            if parent in term_set:
                parents[current].append(parent)
    return {term: parents.get(term, []) for term in terms}


def propagate(scores, terms, parents):
    index = {term: i for i, term in enumerate(terms)}
    state = {}

    def visit(term):
        if state.get(term) == 1:
            raise ValueError("Cycle in GO DAG")
        if state.get(term) == 2:
            return
        state[term] = 1
        for parent in parents[term]:
            visit(parent)
        state[term] = 2
        order.append(term)

    order = []
    for term in terms:
        visit(term)
    result = scores.copy()
    for child in reversed(order):
        for parent in parents[child]:
            result[:, index[parent]] = np.maximum(result[:, index[parent]],
                                                   result[:, index[child]])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=EXAMPLE / "output")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = EXAMPLE / "inputs"
    indices = json.loads((ROOT / "DeepAMIGO/human/HUMAN_target_indices.json").read_text())["MF"]
    terms = json.loads((ROOT / "DeepAMIGO/human/HUMAN_target_go_classes.json").read_text())["MF"]
    assert len(indices) == len(terms) == 314
    model = DeepAMIGO(num_classes=1244).to(device)
    checkpoint = MODEL_DIR / "checkpoints_apa_MF/best_model.pth"
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    names, scores, targets, graph_rows = [], [], [], []
    for record in SeqIO.parse(inputs / "proteins.fasta", "fasta"):
        name, sequence = record.id, str(record.seq)
        features = torch.load(inputs / f"{name}_features.pt", map_location="cpu",
                              weights_only=True)
        length = len(sequence)
        assert all(features[key].shape[0] == length for key in
                   ("x_esm2", "x_protT5", "x_onehot"))
        edges, edge_attr = graph_from_pdb(inputs / f"{name}.pdb", length)
        data = Data(x_esm2=features["x_esm2"].float(),
                    x_protT5=features["x_protT5"].float(),
                    x_onehot=features["x_onehot"].float(),
                    edge_index=edges, edge_attr=edge_attr,
                    batch=torch.zeros(length, dtype=torch.long)).to(device)
        with torch.inference_mode():
            probability = torch.sigmoid(model(data))[0, indices].cpu().numpy()
        names.append(name)
        scores.append(probability)
        targets.append(features["y"].reshape(-1)[indices].numpy())
        graph_rows.append((name, length, edges.shape[1]))
    scores = np.stack(scores)
    targets = np.stack(targets)
    corrected = propagate(scores, terms, parent_map(inputs / "go_mf_subset.obo", terms))
    if not np.isfinite(scores).all() or not np.isfinite(corrected).all():
        raise ValueError("Non-finite prediction score")
    if np.any(corrected < scores):
        raise ValueError("DAG propagation reduced a score")
    metrics = {"raw": compute_cafa_metrics(targets, scores),
               "dag": compute_cafa_metrics(targets, corrected)}
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with (args.output / "predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["protein", "go_term", "target", "raw_score", "dag_score"])
        for i, name in enumerate(names):
            for j, term in enumerate(terms):
                writer.writerow([name, term, int(targets[i, j]),
                                 float(scores[i, j]), float(corrected[i, j])])
    with (args.output / "graphs.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["protein", "residues", "edges"])
        writer.writerows(graph_rows)
    print(json.dumps({"proteins": names, "device": str(device), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
