#!/usr/bin/env python3
"""Config Validator for WaveCap-SDR"""
import argparse
import sys
from pathlib import Path
import yaml

def validate_config(config_path):
    """Validate wavecapsdr.yaml"""
    print(f"Validating: {config_path}")

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"✗ YAML Syntax Error: {e}")
        return 1
    except Exception as e:
        print(f"✗ Error reading file: {e}")
        return 1

    errors = []
    warnings = []

    # Validate server section
    if 'server' not in config:
        errors.append("Missing 'server' section")
    else:
        if 'port' not in config['server']:
            errors.append("Missing server.port")
        elif not isinstance(config['server']['port'], int):
            errors.append("server.port must be integer")

    # Validate device section
    if 'device' not in config:
        errors.append("Missing 'device' section")
    else:
        if 'driver' not in config['device']:
            errors.append("Missing device.driver")

    # Validate presets
    if 'presets' in config:
        for name, preset in config['presets'].items():
            if 'center_hz' not in preset:
                warnings.append(f"Preset '{name}' missing center_hz")
            if 'sample_rate' not in preset:
                warnings.append(f"Preset '{name}' missing sample_rate")

    # Validate recipes
    if 'recipes' in config:
        for name, recipe in config['recipes'].items():
            if 'capture' not in recipe:
                errors.append(f"Recipe '{name}' missing capture")
            if 'channels' not in recipe:
                warnings.append(f"Recipe '{name}' has no channels")

    # Print results
    print()
    if errors:
        print("ERRORS:")
        for e in errors:
            print(f"  ✗ {e}")
    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(f"  ⚠ {w}")

    if not errors and not warnings:
        print("✓ Config is valid!")
        return 0
    elif errors:
        print(f"\n✗ Validation failed with {len(errors)} error(s)")
        return 1
    else:
        print(f"\n✓ Config valid (with {len(warnings)} warning(s))")
        return 0

def main():
    parser = argparse.ArgumentParser(description='Validate wavecapsdr.yaml')
    parser.add_argument('--config', default='backend/config/wavecapsdr.yaml')
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"✗ Config not found: {config_path}")
        return 1

    return validate_config(config_path)

if __name__ == '__main__':
    sys.exit(main())
