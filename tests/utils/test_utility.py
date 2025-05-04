import os
import sys
import unittest

# Add the parent directory to the path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
)

from dlio_benchmark.utils.utility import format_data_size


class TestFormatDataSize(unittest.TestCase):
    def test_base_conversions(self):
        """Test basic unit conversions"""
        # Basic conversions
        self.assertEqual(format_data_size(1024, "B"), "1.0000 KB")
        self.assertEqual(format_data_size(1, "KB"), "1.0000 KB")
        self.assertEqual(format_data_size(1024, "KB"), "1.0000 MB")
        self.assertEqual(format_data_size(1024, "MB"), "1.0000 GB")
        self.assertEqual(format_data_size(1024, "GB"), "1.0000 TB")
        self.assertEqual(format_data_size(1024, "TB"), "1.0000 PB")

    def test_decimal_places(self):
        """Test formatting with different decimal places"""
        # Testing with different decimal places
        self.assertEqual(format_data_size(1536, "KB", decimal_places=0), "2 MB")
        self.assertEqual(format_data_size(1536, "KB", decimal_places=1), "1.5 MB")
        self.assertEqual(format_data_size(1536, "KB", decimal_places=2), "1.50 MB")
        self.assertEqual(format_data_size(1536, "KB", decimal_places=3), "1.500 MB")

    def test_small_values(self):
        """Test handling of small values"""
        # Small values
        self.assertEqual(format_data_size(0.5, "MB"), "512.0000 KB")
        self.assertEqual(format_data_size(0.0003, "GB"), "314.5728 KB")
        self.assertEqual(format_data_size(0.00002, "TB"), "20.9715 MB")

    def test_zero_values(self):
        """Test handling of zero values"""
        # Zero value
        self.assertEqual(format_data_size(0, "KB"), "0.0000 B")
        self.assertEqual(format_data_size(0, "MB"), "0.0000 B")
        self.assertEqual(format_data_size(0, "GB"), "0.0000 B")

    def test_exact_boundary_values(self):
        """Test values at unit boundaries"""
        # Exact boundary values
        self.assertEqual(format_data_size(1023, "B"), "1023.0000 B")
        self.assertEqual(format_data_size(1024.1, "B"), "1.0001 KB")

    def test_invalid_unit(self):
        """Test handling of invalid units"""
        # Testing invalid unit
        with self.assertRaises(ValueError):
            format_data_size(100, "XB")


if __name__ == "__main__":
    unittest.main()
