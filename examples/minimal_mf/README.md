# Three-protein MF workflow example

This example demonstrates the workflow from three protein sequences and PDB structures to GO metric output. Run all commands from the **repository root**. It uses the released Human MF `best_model.pth` checkpoint and the repository's fixed 1,244-output label mapping.

## Inputs

- `inputs/proteins.fasta`: sequences for A0A075B6T6, A0A0A6YYK7, and A0A0B4J1X5.
- `inputs/<ID>.pdb`: the corresponding single-protein structure files used for graph construction.
- `inputs/<ID>_features.pt`: ready-to-use ESM-2 (2,560 dimensions), ProtT5 (1,024 dimensions), one-hot (21 dimensions), and 1,244-position label tensors. The script loads these features before inference.
- `inputs/go_mf_subset.obo`: the MF terms selected from the project's 2025-10-10 `go.obo` snapshot, retaining their original `is_a` and `part_of` records. The fixed MF label list is `DeepAMIGO/human/HUMAN_target_go_classes.json`.
- `DeepAMIGO/human/src_model/checkpoints_apa_MF/best_model.pth`: released MF checkpoint, SHA-256 `4f4335e09ce68f131a766570e07b8bb998fd686582a0a44c0d21fa2df924278b`.

The example inputs were selected from the project's Human test graph objects and corresponding structure files. The labels and embeddings were saved from those graph objects. The script builds graph edges and 16-dimensional RBF edge features from the supplied PDB files using Cα coordinates from the first model and first chain.

## Install and run

The example was verified with Python 3.12, PyTorch 2.5.1, PyTorch Geometric 2.6.1, NumPy 2.1.3, SciPy 1.16.0, scikit-learn 1.7.0, and Biopython 1.85 on CPU. Install a mutually compatible PyTorch and PyTorch Geometric pair for your platform, then:

```bash
python -m pip install torch torch_geometric
python -m pip install numpy scipy scikit-learn biopython
python examples/minimal_mf/run.py
```

`run.py` writes `output/graphs.csv` (node and edge counts), `output/predictions.csv` (target, raw score, DAG score per MF term and protein), and `output/metrics.json` (raw and DAG metrics). To write elsewhere, pass `--output PATH`.

The DAG step propagates each child score to listed parents within the **fixed 314-term MF output space**. The supplied ground-truth labels are used as stored. The released `src_model/metrics.py` evaluates proteins with positive MF labels and scans thresholds from 0 to 1 in 0.01 steps. In this implementation, the `AUPR` and `AUPR_Macro` output keys contain the same protein-level precision–recall integral. AUC is evaluated for terms with both positive and negative proteins.

The metrics report the output of this three-protein workflow demonstration. The manuscript reports Human benchmark performance using the complete test split and its specified evaluation protocol.
