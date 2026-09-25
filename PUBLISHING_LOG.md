# Publication log

## 2026-09-25 21:32:50 UTC — initial public publication preparation

Target: `haideraqeeb-intellifai/fast-h3-v2` (public), requested by the project owner.
Source: local `fast-h3-v2` benchmark workspace; it had no top-level Git revision. Vendored FastVideo source is based on `9b0e57fe4b3a8c112ff24cc22f752eb0608ccfab` from `hao-ai-lab/FastVideo`, with the CPU-load and decoder-no-copyback patches included. Model revision: `3da2ddfe1954d9cda4c05b643dc0f26007a655c5`.

Inventory: 2945 original eligible files and symlinks, plus publication documentation, model/provenance manifests and the configuration selector. Includes full local FastVideo source and resources, all three text-to-audio-video benchmark runs, original executed scripts, prompts, logs, package inventory, model metadata/tokenizers/licenses, output media, HTML reports, earlier attempts and the separate Nsight/ComfyUI profiling archive. Optional upstream submodules retain their exact gitlinks through the root `.gitmodules`.

Intentional exclusions: 32 model weight shards (147,835,483,860 bytes), documented in `MODEL_FILES.json` and `MODEL_REFERENCES.md`; installed `.venv`, downloaded bootstrap tools, uv/Hugging Face caches, Python bytecode, and local Git/tool metadata. No eligible media, profiling traces or model metadata were omitted because of size or ignore patterns. The complete inventory and exclusion list are in `PUBLICATION_INVENTORY.json`.

Publication changes: model links made relative; root runners record pinned upstream provenance and validate the required decoder policy; report builders resolve saved videos inside the clone. Exact historical executed scripts and timing evidence are preserved. The bundled runtime defaults to no-copyback; the selector restores original copy-back for the eight-step measurement configuration. Third-party licenses and notices remain in their original directories. No license was invented for the benchmark-specific work.

Validation: decoder switching round-trip and wrong-policy rejection passed; root Python entrypoints parsed; original executed-runner SHA-256 checks passed; recorded measured times match 183.368 s and 293.715 s; all three report pages' local asset links resolve; both FastH3 runs' warm-up and measured audio/video decode fully. Credential-pattern scan of 2,929 regular source/resource files, including binaries and decompressed gzip artifacts, found no token/key/signature patterns. Assignment-pattern candidates were reviewed as upstream placeholders/test fixtures. GPU inference was not rerun for publication.

Large non-weight resources use Git LFS. GitHub's [large-file rules](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github) and [LFS per-file limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage) were checked on the publication date. The largest object is the 1,988,030,464-byte Nsight SQLite trace.

GitHub initially rejected two vendored upstream blobs as possible Mistral keys. Review found their shared 32-character match was the public Python model class name `Mistral3ForConditionalGeneration`, defined in `fastvideo/models/encoders/mistral3.py` and used by the registry and Flux2 parity test. GitHub accepted the specific `false_positive` resolution through its push-protection API. Source was preserved and repository secret protection was not disabled.
