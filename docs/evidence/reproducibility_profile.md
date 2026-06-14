# Reproducibility Profile

The CR-3 baseline environment is:

- Python 3.13.1
- scikit-learn 1.6.1
- NumPy 2.2.6
- pandas 2.2.3
- matplotlib 3.10.1

The canonical correctness run command is:

```powershell
python scripts/train_credit_risk_model.py --data-path notebook\data\lending_club_accepted_loans.csv --sample-rows 100000 --split-strategy temporal
```

Generated provenance records the parent commit, working-tree state, training
script checksum, dependency versions, sanitized command, dataset fingerprint,
creation timestamp, and artifact checksums. Raw data remains local and ignored.

The committed CR-2 artifacts were generated from commit
`a7ebb5b4896a17da39771850d6a3534109a1afc4`. CR-3 artifacts should be interpreted
using their own `reports/model_validation/provenance.json` record.
