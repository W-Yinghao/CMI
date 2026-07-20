# C84A PM Realized-Policy-Use Addendum

## Status

This is an additive PM interpretation of the frozen C84A/C84S compact result.
It does not replace C84A, recompute a statistic, or alter the immutable C84S
gate `C84-D_external_dataset_source_panel_seed_level_or_target_heterogeneous`
or frontier tag `C84-L4`.

## MaNo And The Fixed B1 Action

The frozen C84A method-context audit shows that U11/MaNo selected an action
with exactly the same held utility and regret as B1/ERM in:

| Dataset | Exact B1-equivalent contexts | Total contexts | Fraction |
|---|---:|---:|---:|
| Lee2019_MI | 175 | 176 | 0.9943181818181818 |
| Cho2017 | 160 | 160 | 1.0 |
| PhysionetMI | 607 | 608 | 0.9983552631578947 |

In Cho, MaNo's realized action map is therefore exactly the fixed B1 action
map on every registered context. The registered MaNo formula still qualifies
under frozen Q1 and Q2, and Cho's frozen category remains A. That qualification
does **not** establish incremental realized decision value from target-unlabeled
information over B1. It also does not establish that the target-unlabeled
experiment contains no information: a different policy could use the same
experiment differently.

The Lee and Physionet results are near-collapse observations, not exact
almost-sure theorems about their populations. They are descriptive properties
of the frozen context field.

## Label And Tail Boundaries

- COTT passing Q2 in all three cohorts does not imply that a qualified
  one-label frontier exists. Q2 compares COTT with the registered Q0 B=1
  policy under its own compound gate; the frontier has separate source-relative
  and larger-budget closure requirements.
- `C84-L4` means at least one registered primary dataset lacks a qualifying
  budget under the fixed Q0 policy and robust closure. It does not mean target
  labels carry no information.
- Comparisons among COTT, MaNo, B1, and Q0 are frozen-policy comparisons, not
  Blackwell comparisons of unrestricted statistical experiments.

## Claim Contract

Supported: Cho MaNo has a frozen Q1/Q2 pass and an exactly B1-equivalent
realized action map on all 160 contexts.

Not supported: MaNo provides incremental target-unlabeled information value in
Cho; labels are less informative than unlabeled outputs; C84-L4 means labels
have no value; or the C84 taxonomy should be changed.

Source: `oaci/reports/c84a_tables/mano_cross_dataset_decision_profile.csv`, a
compact post-C84S table. No EEG, label view, candidate array, selector, Q0,
inference, training, forward, checkpoint, or oracle object was accessed.
