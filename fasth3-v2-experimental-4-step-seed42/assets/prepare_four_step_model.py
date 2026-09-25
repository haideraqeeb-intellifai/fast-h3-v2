"""Create an experimental schedule view without modifying the downloaded model."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'models/FastH3-8-Step-V2'
DEST = ROOT / 'models/FastH3-V2-experimental-4-step'
DEST.mkdir(exist_ok=True)
for source in SOURCE.iterdir():
    if source.name in ('fastvideo_inference.json', '.cache'):
        continue
    target = DEST / source.name
    if target.is_symlink():
        assert target.resolve() == source.resolve(), target
    else:
        target.symlink_to(source, target_is_directory=source.is_dir())
original = SOURCE / 'fastvideo_inference.json'
contract = json.loads(original.read_text())
assert contract['dmd_denoising_steps'] == [999, 874, 749, 624, 500, 375, 250, 125]
contract.update(dmd_denoising_steps=[999, 749, 500, 250], num_inference_steps=5, transformer_forwards=4)
contract['experiment'] = {
    'description': 'Unvalidated four-forward subsampling of the V2 eight-forward trained schedule; original weights unchanged.',
    'original_schedule': [999, 874, 749, 624, 500, 375, 250, 125],
    'original_contract_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
}
target = DEST / 'fastvideo_inference.json'
contents = json.dumps(contract, indent=2) + '\n'
if target.exists():
    assert target.read_text() == contents, 'Existing experimental contract differs; refusing to overwrite.'
else:
    target.write_text(contents)
print(DEST)
