#!/usr/bin/env python3
"""
Quick patch to fix slow protein creation

Run this to automatically fix the slow PDB fetch issue:
    python3 QUICK_FIX_PATCH.py
"""

import os
import shutil
from pathlib import Path

def apply_quick_fix():
    """Apply the quick fix to proteins.py"""

    proteins_file = Path("app/api/proteins.py")

    if not proteins_file.exists():
        print("❌ Error: app/api/proteins.py not found")
        print("   Run this script from the backend/ directory")
        return False

    # Backup original
    backup_file = Path("app/api/proteins_BACKUP.py")
    if not backup_file.exists():
        print("📋 Creating backup: proteins_BACKUP.py")
        shutil.copy(proteins_file, backup_file)
    else:
        print("✓ Backup already exists: proteins_BACKUP.py")

    # Read original
    with open(proteins_file, 'r') as f:
        content = f.read()

    # Check if already patched
    if "# QUICK FIX APPLIED" in content:
        print("✓ Already patched! No changes needed.")
        return True

    # Find and replace the slow PDB fetch block
    old_code = """    # Try to fetch PDB file if the protein name looks like a PDB ID
    pdb_fetch = PDBFetchService()
    if pdb_fetch.is_valid_pdb_id(protein.name):
        try:
            pdb_path = pdb_fetch.fetch_pdb(protein.name, protein_dir)
            db_protein.pdb_file = pdb_path
            db.commit()
            print(f"Successfully fetched PDB file for {protein.name}")
        except Exception as e:
            print(f"Could not fetch PDB file for {protein.name}: {e}")
            # Continue without PDB - not a critical error"""

    new_code = """    # QUICK FIX APPLIED: PDB fetch disabled to speed up protein creation
    # Users can upload PDB files directly via the upload endpoint
    # To re-enable: see proteins_BACKUP.py
    pass"""

    if old_code in content:
        content = content.replace(old_code, new_code)
        print("✓ Applied patch: Disabled PDB fetch")
    else:
        print("⚠️  Warning: Could not find exact match for PDB fetch code")
        print("   File may have been modified. Check manually.")
        return False

    # Write patched file
    with open(proteins_file, 'w') as f:
        f.write(content)

    print("✅ Patch applied successfully!")
    print("")
    print("Next steps:")
    print("  1. Restart the backend server")
    print("  2. Test adding a protein - should be instant!")
    print("")
    print("To revert:")
    print("  cp app/api/proteins_BACKUP.py app/api/proteins.py")
    print("")

    return True


if __name__ == "__main__":
    print("")
    print("=" * 50)
    print(" Quick Fix: Slow Protein Creation")
    print("=" * 50)
    print("")

    success = apply_quick_fix()

    if success:
        exit(0)
    else:
        exit(1)
