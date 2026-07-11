# P9 Accepted Seed 1/2 Shard Evidence

This directory contains the eight accepted seed-by-shard result CSVs, summaries, and Slurm stdout/stderr files used by the final merge.

- Result CSV and summary bytes are copied unchanged from the repository-external run cache.
- Text logs have only trailing horizontal whitespace removed for Git whitespace validation.
- Every committed input checksum is recorded below and in the seeds-1/2 summary JSON.

| shard | job | partition | GPU | rows | result sha256 | summary sha256 | stdout sha256 | stderr sha256 |
|---|---:|---|---|---:|---|---|---|---|
| seed1_shard0 | 892464 | V100 | Tesla V100-PCIE-16GB | 116 | `e1f269aff2de105ce75689e6a864e81cd4d1ca9efc70dba75b018b8d880d515e` | `fcf85d1317ee9e55dbd46641a88045c7937bb4df899d1368e711d0808020ee8c` | `0024861e465c581aeb81f94a3ebe71b29aef1851e5ffbf978bd38987f03b404e` | `73ab8fa6eeacd7956bc5025e0c7f8b56257e61a2f62e3c24c73ad99d3cd8b28e` |
| seed1_shard1 | 892465 | V100 | Tesla V100S-PCIE-32GB | 116 | `91cbd26498c2c53540080d0113d0aef6e2e9ea7caafbdc18da9545e8475f6e44` | `ce6b942732e27985a2cb505bb6f6ea787a85821dba905121e62f2295fd50a385` | `dab264ff5d28609b99262ac2f9533ad53c1c47456e7fb61c779c9a579ce6201a` | `73ab8fa6eeacd7956bc5025e0c7f8b56257e61a2f62e3c24c73ad99d3cd8b28e` |
| seed1_shard2 | 892466 | V100 | Tesla V100S-PCIE-32GB | 116 | `cf8255c606aaa62f2c9e075c0f415f917ae87f9e70819129db4189f23818def2` | `6728d9cd8383e2669434fb816c6bdfdb21b52f4a59438702e64e21f1ba8876ea` | `443690c81c9dc443dc778817b6920481e6072be77e2643f53bab6f35ed3c1f62` | `73ab8fa6eeacd7956bc5025e0c7f8b56257e61a2f62e3c24c73ad99d3cd8b28e` |
| seed1_shard3 | 892467 | V100 | Tesla V100S-PCIE-32GB | 112 | `ca822dbfea63c9310a7803a908bfbcc904e169ab5d1b23c8e97b6bfd11560d7e` | `4185de75af4a7c6d0f675f78e1fd73567d4fb865eaa8b12c6ea4ae8c323a3729` | `e7b85bc002e05d2f3cee3ceb3c340a52982ae6a22afa05fe8de4bf4ec02cfcb2` | `cace1071573ef725e762e8856427937617c8c67e5d0784c45c0e854e56a803a8` |
| seed2_shard0 | 892842 | V100 | Tesla V100S-PCIE-32GB | 116 | `201db54ca08d8c805e5266ef9020da8e3c9a3bc5b66ac3261e248226ae4c0004` | `005477284ada9957d497fdc2012151b46f297bc456074e0a26ee364d7db644c8` | `ac14ae7fa272abfa6ca95859d5bce3c52f023f77bb0d11e4ad4548550e3dbf2e` | `73ab8fa6eeacd7956bc5025e0c7f8b56257e61a2f62e3c24c73ad99d3cd8b28e` |
| seed2_shard1 | 892883 | A40 | NVIDIA A40 | 116 | `98ca21b317eecb16163d2a23301e17d7101c9b6de31d9d04f8f749556f31cb87` | `30a5bd2493010b878fd2781a9694ec4acf24610bbd0fb59cd0d5f56001f4d65b` | `a3dac88c4de2c3c17c2e079b3e09352e26a03c7bc3362c362b171d29cf611be2` | `73ab8fa6eeacd7956bc5025e0c7f8b56257e61a2f62e3c24c73ad99d3cd8b28e` |
| seed2_shard2 | 892957 | V100 | Tesla V100S-PCIE-32GB | 116 | `f79b02fcb122cf0cf9c0134d9003fb727a47c5ffaa358c4994213d1ad02378dc` | `79a983f947ecc81e614b109bdbb054e177f7db12e31c3a63b9f019ea3db7163f` | `f8d096663a6b8abe4b95ce07233767f9994e78630c3a5f99cd390277f355ecba` | `73ab8fa6eeacd7956bc5025e0c7f8b56257e61a2f62e3c24c73ad99d3cd8b28e` |
| seed2_shard3 | 892389 | A100 | NVIDIA A100-SXM4-40GB | 112 | `b3f4b1bf77125f816d8687afe3603ac0cdad97f22d7211b77b43c0d6578c9465` | `bcf6e5a7d52a8d199151c6967fe13f2a4f2c33a752f1efd83740285b57b39f66` | `e013615ef5421eb29c8bf37466c4771fa427c2e3c95a1258645fa759860e8674` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

## Internal Validation Review

- All eight inputs pass provenance, log, row-count, split, and leakage gates.
- The excluded P100 launch is stored separately and contributes no rows.
