# Archived original notebooks

These are the v0.1 notebooks. They are kept here for git history and so the
"before/after" of the refactor is visible to anyone reading the repo.

They have several known issues that motivated the rewrite — none is fixed in
these archived copies; see the new notebooks under `notebooks/` for the clean
versions:

1. **`preprocess_data.ipynb` looks for files that do not exist** — it scans for
   `*_annotations.csv`, but the AutoRPT corpus on disk only ships `.ton` (ToBI
   labels) and `.TextGrid` files. The pickle that the rest of the pipeline
   loads was apparently produced from an external CSV that is no longer in the
   repo. The new pipeline parses `.ton` directly so it is reproducible end to
   end.
2. **The classical-vs-neural comparison was apples-to-oranges** — classical
   models were evaluated on per-frame predictions (positive rate ~17%
   prominence / ~5% boundary), and the neural models were evaluated on
   per-window predictions after a re-labeling step that pushed the positive
   rate to ~67% / ~23%. The rewritten code reports both tasks for both model
   families.
3. **The split was based on file order, not speaker identity** — speakers were
   not guaranteed to be disjoint across train, validation, and test.
4. **The neural scaler was fitted on the full corpus**, including validation
   and test features. The rewrite fits normalization on training data only.
5. **Model objects and checkpoint filenames were reused** — the classical test
   cell could evaluate a boundary-fitted Random Forest as the prominence model,
   and the neural notebook overwrote the CNN checkpoint with the RNN. The
   rewrite keeps independent estimators and checkpoint paths.
