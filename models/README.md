# Model ports

Create one directory per model using the layout in
[`docs/PORTING_CONTRACT.md`](../docs/PORTING_CONTRACT.md). Model weights are
external artifacts; this repository stores pinned download URLs, sizes and
SHA-256 digests so `tools/model_assets.py download` can reproduce them.

Recommended commands:

```text
python tools/run_conversion.py --spec models/<id>/conversion.json
python tools/model_assets.py download --manifest models/<id>/assets.json --root <model-dir>
python tools/model_assets.py verify --manifest models/<id>/assets.json --root <model-dir>
python tools/compare_outputs.py text --reference ref.txt --actual ncnn.txt --receipt parity.json
```
