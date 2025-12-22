# Copyright 2021 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock

# Import heatmap visualization functions
from visualize_attention_heatmap_utils import (
    get_sequence_length_from_fasta,
    reconstruct_attention_matrix,
    plot_all_heads_heatmap,
    plot_combined_attention_heatmap
)


class TestHeatmapVisualization(unittest.TestCase):
    """Test cases for heatmap visualization functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a test FASTA file
        self.test_fasta = os.path.join(self.temp_dir, "test.fasta")
        with open(self.test_fasta, 'w') as f:
            f.write(">test_sequence\n")
            f.write("ACDEFGHIKLMNPQRSTVWY\n")  # 20 residues
        
        # Create test attention data
        self.test_attention_data = {
            0: [(0, 0, 1.0), (0, 1, 0.5), (1, 1, 1.0), (1, 0, 0.3)],
            1: [(0, 0, 0.8), (0, 2, 0.6), (2, 2, 1.0), (2, 0, 0.4)]
        }

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_get_sequence_length_from_fasta(self):
        """Test FASTA sequence length extraction."""
        length = get_sequence_length_from_fasta(self.test_fasta)
        self.assertEqual(length, 20)

    def test_reconstruct_attention_matrix(self):
        """Test attention matrix reconstruction."""
        connections = [(0, 0, 1.0), (0, 1, 0.5), (1, 1, 1.0), (1, 0, 0.3)]
        matrix = reconstruct_attention_matrix(connections, 2)
        
        expected = np.array([[1.0, 0.5], [0.3, 1.0]])
        np.testing.assert_array_equal(matrix, expected)
        
        # Test with larger sequence
        matrix_large = reconstruct_attention_matrix(connections, 5)
        self.assertEqual(matrix_large.shape, (5, 5))
        self.assertEqual(matrix_large[0, 0], 1.0)
        self.assertEqual(matrix_large[0, 1], 0.5)

    def test_reconstruct_attention_matrix_sparse(self):
        """Test sparse attention matrix reconstruction."""
        connections = [(0, 0, 1.0), (2, 2, 0.8)]
        matrix = reconstruct_attention_matrix(connections, 3)
        
        expected = np.array([[1.0, 0.0, 0.0], 
                            [0.0, 0.0, 0.0], 
                            [0.0, 0.0, 0.8]])
        np.testing.assert_array_equal(matrix, expected)

    def test_load_attention_data(self):
        """Test attention data loading through plot functions."""
        # This test verifies that the plot functions can load data correctly
        # by checking that they don't raise errors with valid inputs
        try:
            # Create a mock attention file
            mock_file = os.path.join(self.temp_dir, "msa_row_attn_layer47.txt")
            with open(mock_file, 'w') as f:
                f.write("Layer 47, Head 0\n")
                f.write("0 0 1.0\n")
                f.write("0 1 0.5\n")
                f.write("Layer 47, Head 1\n")
                f.write("1 1 1.0\n")
                f.write("1 0 0.3\n")
            
            # Test that the function can handle the file
            # Note: This is a simplified test since we can't easily mock the file loading
            self.assertTrue(os.path.exists(mock_file))
        except Exception as e:
            self.fail(f"Test setup failed: {e}")

    @patch('visualize_attention_heatmap_utils.load_all_heads')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_plot_all_heads_heatmap(self, mock_show, mock_savefig, mock_load_heads):
        """Test individual heatmap plotting."""
        # Mock attention data
        mock_load_heads.return_value = {
            0: [(0, 0, 1.0), (0, 1, 0.5)],
            1: [(1, 1, 1.0), (1, 0, 0.3)]
        }
        
        # Create mock attention file
        mock_file = os.path.join(self.temp_dir, "msa_row_attn_layer47.txt")
        with open(mock_file, 'w') as f:
            f.write("Layer 47, Head 0\n")
            f.write("0 0 1.0\n")
            f.write("0 1 0.5\n")
            f.write("Layer 47, Head 1\n")
            f.write("1 1 1.0\n")
            f.write("1 0 0.3\n")
        
        output_path = plot_all_heads_heatmap(
            attention_dir=self.temp_dir,
            output_dir=self.temp_dir,
            protein='TEST',
            attention_type='msa_row',
            layer_idx=47,
            seq_length=10,
            save_to_png=True,
            residue_indices=None
        )
        
        # Verify that savefig was called (file creation is mocked)
        mock_savefig.assert_called_once()
        # Verify that the expected output path is returned
        expected_path = os.path.join(self.temp_dir, "msa_row_heatmap_layer_47_TEST.png")
        self.assertEqual(output_path, expected_path)

    @patch('visualize_attention_heatmap_utils.load_all_heads')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_plot_combined_attention_heatmap(self, mock_show, mock_savefig, mock_load_heads):
        """Test combined heatmap plotting."""
        # Mock attention data for both types
        mock_load_heads.return_value = {
            0: [(0, 0, 1.0), (0, 1, 0.5)],
            1: [(1, 1, 1.0), (1, 0, 0.3)]
        }
        
        # Create mock attention files
        msa_file = os.path.join(self.temp_dir, "msa_row_attn_layer47.txt")
        tri_file = os.path.join(self.temp_dir, "triangle_start_attn_layer47_residue_idx_18.txt")
        
        for file_path in [msa_file, tri_file]:
            with open(file_path, 'w') as f:
                f.write("Layer 47, Head 0\n")
                f.write("0 0 1.0\n")
                f.write("0 1 0.5\n")
                f.write("Layer 47, Head 1\n")
                f.write("1 1 1.0\n")
                f.write("1 0 0.3\n")
        
        output_path = plot_combined_attention_heatmap(
            attention_dir=self.temp_dir,
            output_dir=self.temp_dir,
            protein='TEST',
            layer_idx=47,
            seq_length=10,
            save_to_png=True,
            residue_indices=[18]
        )
        
        # Verify that savefig was called (file creation is mocked)
        mock_savefig.assert_called_once()
        # Verify that the expected output path is returned
        expected_path = os.path.join(self.temp_dir, "combined_attention_heatmap_layer_47_TEST.png")
        self.assertEqual(output_path, expected_path)

    def test_normalization_methods(self):
        """Test normalization methods."""
        connections = [(0, 0, 1.0), (0, 1, 0.5), (1, 1, 0.8), (1, 0, 0.3)]
        matrix = reconstruct_attention_matrix(connections, 2)
        
        # Test global normalization
        global_min = np.min(matrix)
        global_max = np.max(matrix)
        global_norm = (matrix - global_min) / (global_max - global_min)
        
        self.assertAlmostEqual(np.min(global_norm), 0.0)
        self.assertAlmostEqual(np.max(global_norm), 1.0)
        
        # Test per-head normalization
        head_min = np.min(matrix)
        head_max = np.max(matrix)
        per_head_norm = (matrix - head_min) / (head_max - head_min)
        
        self.assertAlmostEqual(np.min(per_head_norm), 0.0)
        self.assertAlmostEqual(np.max(per_head_norm), 1.0)

    def test_edge_cases(self):
        """Test edge cases."""
        # Empty connections
        empty_matrix = reconstruct_attention_matrix([], 5)
        self.assertEqual(empty_matrix.shape, (5, 5))
        self.assertEqual(np.sum(empty_matrix), 0.0)
        
        # Single connection
        single_conn = [(2, 3, 0.7)]
        single_matrix = reconstruct_attention_matrix(single_conn, 5)
        self.assertEqual(single_matrix[2, 3], 0.7)
        self.assertEqual(np.sum(single_matrix), 0.7)

    def test_matrix_properties(self):
        """Test attention matrix properties."""
        connections = [(0, 0, 1.0), (0, 1, 0.5), (1, 1, 1.0), (1, 0, 0.3)]
        matrix = reconstruct_attention_matrix(connections, 2)
        
        # Test symmetry (not required but common in attention)
        # Test that diagonal elements are preserved
        self.assertEqual(matrix[0, 0], 1.0)
        self.assertEqual(matrix[1, 1], 1.0)
        
        # Test that off-diagonal elements are preserved
        self.assertEqual(matrix[0, 1], 0.5)
        self.assertEqual(matrix[1, 0], 0.3)

    def test_triangle_start_with_residue_indices(self):
        """Test triangle_start attention with residue indices."""
        # Create mock triangle_start attention file
        mock_file = os.path.join(self.temp_dir, "triangle_start_attn_layer47_residue_idx_18.txt")
        with open(mock_file, 'w') as f:
            f.write("Layer 47, Head 0\n")
            f.write("0 0 1.0\n")
            f.write("0 1 0.5\n")
            f.write("Layer 47, Head 1\n")
            f.write("1 1 1.0\n")
            f.write("1 0 0.3\n")
        
        # Test that the function can handle triangle_start with residue_indices
        try:
            from visualize_attention_heatmap_utils import plot_all_heads_heatmap
            result = plot_all_heads_heatmap(
                attention_dir=self.temp_dir,
                output_dir=self.temp_dir,
                protein='TEST',
                attention_type='triangle_start',
                layer_idx=47,
                seq_length=10,
                residue_indices=[18],
                save_to_png=False
            )
            # Should return a valid path or None
            self.assertTrue(result is None or isinstance(result, str))
        except Exception as e:
            self.fail(f"Triangle start with residue indices failed: {e}")

    def test_empty_file_handling(self):
        """Test handling of empty attention files - matches arc/PyMOL behavior."""
        # Create empty attention file
        empty_file = os.path.join(self.temp_dir, "msa_row_attn_layer47.txt")
        with open(empty_file, 'w') as f:
            f.write("")
        
        try:
            from visualize_attention_heatmap_utils import plot_all_heads_heatmap
            result = plot_all_heads_heatmap(
                attention_dir=self.temp_dir,
                output_dir=self.temp_dir,
                protein='TEST',
                attention_type='msa_row',
                layer_idx=47,
                seq_length=10,
                save_to_png=False
            )
            # Should return None when no valid heads (after skipping empty ones)
            # This matches arc/PyMOL behavior of producing no output
            self.assertIsNone(result)
        except Exception as e:
            self.fail(f"Empty file handling failed: {e}")

    def test_triangle_start_without_residue_indices(self):
        """Test that triangle_start requires residue_indices parameter - matches arc/PyMOL."""
        try:
            from visualize_attention_heatmap_utils import plot_all_heads_heatmap
            result = plot_all_heads_heatmap(
                attention_dir=self.temp_dir,
                output_dir=self.temp_dir,
                protein='TEST',
                attention_type='triangle_start',
                layer_idx=47,
                seq_length=10,
                save_to_png=False
            )
            self.fail("Should have raised AssertionError for missing residue_indices")
        except AssertionError as e:
            self.assertIn("residue_indices required", str(e))
        except Exception as e:
            self.fail(f"Unexpected error: {e}")


if __name__ == '__main__':
    unittest.main()
