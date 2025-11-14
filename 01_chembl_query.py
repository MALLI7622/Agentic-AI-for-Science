#!/usr/bin/env python3
"""
Query ChEMBL for EGFR inhibitors with IC50 < 50nM
"""

import json
import pandas as pd
from bioservices import ChEMBL

print("=" * 80)
print("STEP 1: Querying ChEMBL for EGFR Inhibitors (IC50 < 50nM)")
print("=" * 80)

# Initialize ChEMBL service
chembl = ChEMBL(verbose=False)

# Search for EGFR target
print("\n[1/5] Searching for EGFR target in ChEMBL...")
# Use known EGFR target ID (ChEMBL's canonical human EGFR target)
egfr_target_id = "CHEMBL203"
print(f"   Using EGFR target: {egfr_target_id}")

# Get bioactivities for EGFR with IC50 < 50nM
print(f"\n[2/5] Retrieving bioactivities for {egfr_target_id}...")
print("   Filtering: IC50 < 50nM, standard type = IC50")

# Use requests library directly with ChEMBL REST API
import time
import requests

activities = []
offset = 0
limit = 1000  # Max per page
base_url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"

while True:
    # Query activities with filters
    params = {
        'target_chembl_id': egfr_target_id,
        'standard_type': 'IC50',
        'standard_value__lte': 50,
        'standard_units': 'nM',
        'limit': limit,
        'offset': offset
    }

    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        result = response.json()

        # Debug: Print result type and structure on first iteration
        if offset == 0:
            print(f"   Result type: {type(result)}")
            if isinstance(result, dict):
                print(f"   Result keys: {list(result.keys())[:5]}")
            print(f"   Query URL: {response.url}")

        if isinstance(result, dict) and 'activities' in result:
            batch = result['activities']
            activities.extend(batch)
            print(f"   Retrieved {len(activities)} activities so far...")

            # Check if there are more results
            if len(batch) < limit:
                break
            offset += limit
            time.sleep(0.5)  # Be nice to the API
        else:
            print(f"   No 'activities' key in response. Response type: {type(result)}")
            if isinstance(result, dict):
                print(f"   Available keys: {list(result.keys())}")
            break
    except requests.exceptions.RequestException as e:
        print(f"   Error querying activities: {e}")
        break
    except Exception as e:
        print(f"   Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        break

print(f"   Found {len(activities)} total activities")

# Debug: Check what fields are available
if activities:
    print(f"\n   Sample activity fields: {list(activities[0].keys())[:10]}")

# Extract unique molecules with their best IC50 values
print("\n[3/5] Processing unique molecules...")
molecules_dict = {}
for activity in activities:
    mol_id = activity.get('molecule_chembl_id')
    if mol_id:
        ic50_value = activity.get('standard_value')
        pchembl = activity.get('pchembl_value')

        # Only process if we have valid IC50 data
        if ic50_value is not None:
            try:
                ic50_float = float(ic50_value)
                # Keep the best (lowest) IC50 for each molecule
                if mol_id not in molecules_dict or ic50_float < molecules_dict[mol_id]['ic50_nM']:
                    molecules_dict[mol_id] = {
                        'chembl_id': mol_id,
                        'ic50_nM': ic50_float,
                        'pchembl_value': pchembl,
                    }
            except (ValueError, TypeError):
                continue

print(f"   Extracted {len(molecules_dict)} unique molecules")

if len(molecules_dict) == 0:
    print("\n   ERROR: No valid molecules found. Check the debug output above.")
    print("   This could mean:")
    print("   1. The API query returned no results")
    print("   2. The API response structure is different than expected")
    print("   3. There's a connection issue with the ChEMBL API")
    exit(1)

# Convert to DataFrame
df = pd.DataFrame.from_dict(molecules_dict, orient='index')
print(f"   DataFrame shape: {df.shape}")
print(f"   DataFrame columns: {list(df.columns)}")
df = df.sort_values('ic50_nM')

# Get additional molecular properties including SMILES
print("\n[4/5] Retrieving molecular properties...")
enhanced_molecules = []
count = 0

for idx, row in df.head(100).iterrows():  # Limit to top 100 for API efficiency
    mol_id = row['chembl_id']
    count += 1
    if count % 10 == 0:
        print(f"   Processing molecule {count}/100...")

    try:
        mol_details = chembl.get_molecule(mol_id)

        if mol_details:
            mol_props = mol_details.get('molecule_properties', {})
            mol_structures = mol_details.get('molecule_structures', {})

            enhanced_molecules.append({
                'chembl_id': mol_id,
                'name': mol_details.get('pref_name', 'N/A'),
                'smiles': mol_structures.get('canonical_smiles', 'N/A'),
                'ic50_nM': row['ic50_nM'],
                'pchembl_value': row['pchembl_value'],
                'mw': mol_props.get('full_mwt'),
                'alogp': mol_props.get('alogp'),
                'hba': mol_props.get('hba'),
                'hbd': mol_props.get('hbd'),
                'psa': mol_props.get('psa'),
                'rtb': mol_props.get('rtb'),
                'ro5_violations': mol_props.get('num_ro5_violations'),
            })
        time.sleep(0.2)  # Be nice to the API
    except Exception as e:
        print(f"   Warning: Could not retrieve details for {mol_id}: {e}")
        continue

df_enhanced = pd.DataFrame(enhanced_molecules)
print(f"   Successfully retrieved {len(df_enhanced)} molecules with full details")

# Save results
print("\n[5/5] Saving results...")
df_enhanced.to_csv('chembl_egfr_inhibitors.csv', index=False)
df_enhanced.to_json('chembl_egfr_inhibitors.json', orient='records', indent=2)

print(f"\nSaved {len(df_enhanced)} EGFR inhibitors to:")
print(f"  - chembl_egfr_inhibitors.csv")
print(f"  - chembl_egfr_inhibitors.json")

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print(f"Total compounds retrieved: {len(df_enhanced)}")
print(f"\nIC50 Statistics (nM):")
print(f"  Min:    {df_enhanced['ic50_nM'].min():.2f}")
print(f"  Max:    {df_enhanced['ic50_nM'].max():.2f}")
print(f"  Mean:   {df_enhanced['ic50_nM'].mean():.2f}")
print(f"  Median: {df_enhanced['ic50_nM'].median():.2f}")

print(f"\nMolecular Weight Statistics:")
print(f"  Min:    {df_enhanced['mw'].min():.2f}")
print(f"  Max:    {df_enhanced['mw'].max():.2f}")
print(f"  Mean:   {df_enhanced['mw'].mean():.2f}")

print(f"\nLipinski's Rule of 5 Violations:")
print(f"  Compounds with 0 violations: {len(df_enhanced[df_enhanced['ro5_violations'] == 0])}")
print(f"  Compounds with 1+ violations: {len(df_enhanced[df_enhanced['ro5_violations'] > 0])}")

print("\n" + "=" * 80)
print("Top 10 Most Potent EGFR Inhibitors:")
print("=" * 80)
print(df_enhanced[['chembl_id', 'name', 'ic50_nM', 'mw', 'alogp']].head(10).to_string(index=False))
print("\n")
