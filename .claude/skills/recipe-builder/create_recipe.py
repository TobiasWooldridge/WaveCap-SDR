#!/usr/bin/env python3
"""Recipe Builder for WaveCap-SDR"""
import argparse
import sys
import yaml

def create_recipe(name, center, sample_rate, channels, gain=30):
    """Generate recipe YAML"""
    recipe = {
        'name': name.title(),
        'description': f"Generated recipe for {name}",
        'capture': {
            'center_hz': int(center),
            'sample_rate': int(sample_rate),
            'gain_db': gain,
        },
        'channels': []
    }

    for ch_spec in channels:
        parts = ch_spec.split(':')
        ch_name = parts[0]
        offset = int(parts[1]) if len(parts) > 1 else 0
        recipe['channels'].append({
            'name': ch_name,
            'offset_hz': offset,
            'mode': 'fm',
        })

    return {name: recipe}

def main():
    parser = argparse.ArgumentParser(description='Generate recipe YAML')
    parser.add_argument('--name', required=True)
    parser.add_argument('--center', type=float, required=True)
    parser.add_argument('--sample-rate', type=int, required=True)
    parser.add_argument('--channels', nargs='+', required=True)
    parser.add_argument('--gain', type=int, default=30)

    args = parser.parse_args()

    recipe = create_recipe(args.name, args.center, args.sample_rate, args.channels, args.gain)

    print("\n# Add this to backend/config/wavecapsdr.yaml under 'recipes':\n")
    print(yaml.dump({'recipes': recipe}, default_flow_style=False, sort_keys=False))

    return 0

if __name__ == '__main__':
    sys.exit(main())
