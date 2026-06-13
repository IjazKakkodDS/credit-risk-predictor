# Model Validation Requirements

The following evidence is required before the project can support an elite
technical review:

1. A documented train/test split, including time boundaries or stratification,
   random seeds, leakage controls, and target definition.
2. A class imbalance profile for training and holdout data.
3. A simple baseline model with reproducible configuration and results.
4. A selected champion model with traceable training parameters and artifact
   provenance.
5. A confusion matrix tied to an explicitly stated decision threshold.
6. ROC-AUC with the evaluated population and split identified.
7. PR-AUC, which is especially important when adverse outcomes are uncommon.
8. A calibration curve and an appropriate calibration error summary.
9. A documented threshold policy linked to business costs and review capacity.
10. Analysis of false approval and false rejection trade-offs.
11. Subgroup or segment validation when suitable fields and governance permit
    it.
12. A model card covering intended use, exclusions, data provenance,
    limitations, metrics, ethical considerations, and maintenance expectations.

Metrics must come from a reproducible evaluation run. Plots and reports should
identify the model artifact, preprocessor artifact, dataset version, code
revision, and evaluation command that produced them.
