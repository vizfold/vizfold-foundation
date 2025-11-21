"""
Service for fetching PDB files from RCSB PDB database
"""
import os
import requests


class PDBFetchService:
    """Service to fetch PDB structures from RCSB database"""

    RCSB_BASE_URL = "https://files.rcsb.org/download"

    def fetch_pdb(self, pdb_id: str, output_dir: str) -> str:
        """
        Fetch a PDB file from RCSB database

        Args:
            pdb_id: 4-character PDB ID (e.g., '6KWC')
            output_dir: Directory to save the PDB file

        Returns:
            Path to the downloaded PDB file

        Raises:
            ValueError: If PDB ID is invalid
            requests.HTTPError: If PDB file cannot be downloaded
        """
        # Validate PDB ID format
        pdb_id = pdb_id.upper().strip()
        if len(pdb_id) != 4:
            raise ValueError(f"Invalid PDB ID: {pdb_id}. PDB IDs must be exactly 4 characters.")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Construct download URL
        url = f"{self.RCSB_BASE_URL}/{pdb_id}.pdb"

        # Download PDB file (5 second timeout for faster failures)
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
        except requests.RequestException as e:
            raise requests.HTTPError(
                f"Failed to download PDB {pdb_id} from RCSB database. "
                f"Please check if the PDB ID is correct. Error: {e}"
            )

        # Save to file
        output_path = os.path.join(output_dir, f"{pdb_id}_relaxed.pdb")
        with open(output_path, 'w') as f:
            f.write(response.text)

        return output_path

    def is_valid_pdb_id(self, pdb_id: str) -> bool:
        """
        Check if a string looks like a valid PDB ID

        Args:
            pdb_id: String to check

        Returns:
            True if it matches PDB ID format (4 alphanumeric characters)
        """
        pdb_id = pdb_id.upper().strip()
        return len(pdb_id) == 4 and pdb_id.isalnum()
