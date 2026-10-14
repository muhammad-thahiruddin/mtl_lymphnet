#!/usr/bin/env python
"""Switch the proposed backbone across the ENTIRE framework.

Usage:
    python configs/switch_backbone.py TinyViT-5M
    python configs/switch_backbone.py MobileNetV3-Large
    python configs/switch_backbone.py EfficientNet-B0

This script updates:
  1. src/mtl_lymphnet/models/backbones.py  -- PROPOSED_KEY
  2. src/mtl_lymphnet/config.py            -- default backbone string
  3. configs/*.yaml                        -- backbone: field + comments
  4. notebooks/mtl_lymphnet_runner.ipynb    -- markdown description

After running, all code (ablation, stats, tables, figures) will automatically
use the new proposed backbone at runtime -- no manual edits needed.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

BACKBONES = {
    "TinyViT-5M": "tu-tiny_vit_5m_224.dist_in22k_ft_in1k",
    "MobileNetV3-Large": "tu-mobilenetv3_large_100",
    "EfficientNet-B0": "tu-efficientnet_b0",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def switch_backbone(new_key: str) -> None:
    if new_key not in BACKBONES:
        print(f"ERROR: '{new_key}' not valid. Choose from: {list(BACKBONES.keys())}")
        sys.exit(1)

    new_timm = BACKBONES[new_key]
    variant_names = [k for k in BACKBONES if k != new_key]
    changes: list[str] = []

    # 1. backbones.py
    bp = PROJECT_ROOT / "src" / "mtl_lymphnet" / "models" / "backbones.py"
    text = bp.read_text(encoding="utf-8")
    text = re.sub(r'PROPOSED_KEY = ".*?"', f'PROPOSED_KEY = "{new_key}"', text)
    bp.write_text(text, encoding="utf-8")
    changes.append(f"  backbones.py: PROPOSED_KEY = '{new_key}'")

    # 2. config.py
    cp = PROJECT_ROOT / "src" / "mtl_lymphnet" / "config.py"
    text = cp.read_text(encoding="utf-8")
    text = re.sub(r'backbone: str = "tu-[^"]*"', f'backbone: str = "{new_timm}"', text)
    cp.write_text(text, encoding="utf-8")
    changes.append(f"  config.py: backbone default = '{new_timm}'")

    # 3. YAML configs
    for yaml_file in sorted((PROJECT_ROOT / "configs").glob("*.yaml")):
        text = yaml_file.read_text(encoding="utf-8")
        original = text
        text = re.sub(r'^backbone: tu-.*$', f'backbone: {new_timm}', text, flags=re.MULTILINE)
        for bk in BACKBONES:
            if bk == new_key:
                continue
            text = text.replace(f"{bk} backbone", f"{new_key} backbone")
            text = text.replace(f"{bk} (proposed)", f"{new_key} (proposed)")
        vs = " and ".join(variant_names)
        text = re.sub(
            r'# .+ \(proposed\) is reused from Experiment 1; .+',
            f'# {new_key} (proposed) is reused from Experiment 1; {vs}',
            text,
        )
        if text != original:
            yaml_file.write_text(text, encoding="utf-8")
            changes.append(f"  {yaml_file.name}: updated")

    # 4. Notebook
    nb_path = PROJECT_ROOT / "notebooks" / "mtl_lymphnet_runner.ipynb"
    if nb_path.exists():
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        mod = False
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "markdown":
                continue
            src2 = []
            for line in cell["source"]:
                o = line
                for bk in BACKBONES:
                    if bk == new_key:
                        continue
                    line = line.replace(f"**Backbone:** {bk}", f"**Backbone:** {new_key}")
                    line = line.replace(f"{bk} (data-driven", f"{new_key} (data-driven")
                    line = line.replace(f"{bk} (proposed", f"{new_key} (proposed")
                src2.append(line)
                if line != o:
                    mod = True
            cell["source"] = src2
        if mod:
            nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
            changes.append("  mtl_lymphnet_runner.ipynb: updated")

    print(f"\nSwitched to: {new_key} ({new_timm})")
    print(f"Variants: {variant_names}\n")
    for c in changes:
        print(c)
    print(f"\n{len(changes)} files updated.")
    print("\nIMPORTANT: Delete outputs_v2/experiment_results.pkl and re-run!")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        print(f"Available: {list(BACKBONES.keys())}")
        sys.exit(1)
    switch_backbone(sys.argv[1])
