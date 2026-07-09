# C65 - Checkpoint Storage Provenance

Primary OACI checkout contains no `.pt`, `.pth`, `.ckpt`, or `.safetensors` checkpoint payload, but the mounted `/projects/EEG-foundation-model/yinghao/oaci-loso-seeds012` store contains the recovered frozen checkpoint universe.

Artifact indexes are the hash authority for checkpoint payloads. C65 does not bulk rehash or copy checkpoint files.
