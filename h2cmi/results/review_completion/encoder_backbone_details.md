# Encoder / Backbone Details

Recovered from code and bundle sidecars, not memory.

- MI preprocessing: MOABB MotorImagery binary L/R; deterministic band-pass/resample/window in h2cmi/data/real_eeg.py; per-trial z-score per channel; optional common channel grid for cross-dataset panels.
- Sleep preprocessing: Sleep-EDF Sleep-Cassette only; EEG Fpz-Cz and Pz-Oz; 100 Hz; 30s epochs; W/N1/N2/N3/REM with S3+S4->N3; crop +/-30 min around scored sleep; per-epoch z-score per channel.
- Backbone: H2Encoder: temporal EEGNet-like branch, SPD covariance/log-tangent branch, graph/electrode set branch, fused MLP, split z_c/z_n, optional near-identity canonicalizer.
- Latent dims/defaults: `{'z_c_dim': 32, 'z_n_dim': 16, 'fuse_hidden': 128, 'temporal_filters': 8, 'spd_rank': 8, 'graph_hidden': 16, 'bands_Hz': [[4, 8], [8, 13], [13, 30], [30, 45]]}`
- Source class-conditionals: ClassConditionalDensity Student-t mixture, default one component/class, low-rank+diagonal covariance rank 4, df 8, eig_floor 1e-2.
- Training defaults: Adam via h2cmi/train/trainer.py; defaults lr=1e-3, weight_decay=1e-4, batch_size=64, grad_clip=5, no drop_last. Bundle sidecars record epochs/n_chans/n_train/seed/hash.
- Split protocol: MI W1 LOSO target subject; V2P cross-session source session -> target session; Sleep W2 leave-one-subject/pair protocol with adaptation/evaluation night split.
- Geometry optimization: ClassConditionalTTA diagonal affine by default: z = exp(a)*u + b; EM updates prior and optimizes transform by Adam, em_iters default 20, em_lr 5e-2, trust_region=1, trust_region_b=1, logdet_weight=1, Dirichlet/prior anchor.

Missing fields are listed in `encoder_backbone_details.json`.
