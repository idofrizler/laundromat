"""
End-to-end tests for sock pair detection and matching.

Tests the complete pipeline:
1. SAM3 sock detection (verify 10 socks detected)
2. ResNet feature extraction
3. Pair matching (verify correct pairs based on spatial arrangement)

Test data:
- straight_line/: Socks arranged top-to-bottom, pairs are consecutive (1&2, 3&4, etc.)
- outside_in/: Socks arranged top-to-bottom, pairs are mirrored (1&10, 2&9, etc.)
"""

import pytest
from pathlib import Path

from testing.test_utils import (
    sort_socks_by_y_position,
    get_expected_pairs,
    map_detected_pairs_to_positions,
    normalize_pairs,
    create_result_visualization,
    run_detection_pipeline,
    get_relative_pair_positions,
    check_pair_ordering,
)
from testing.conftest import DATA_DIR

def get_all_test_images():
    """Get all test images with their folder types for parametrization."""
    images = []
    
    # Straight line images
    straight_line_dir = DATA_DIR / "straight_line"
    if straight_line_dir.exists():
        for img in sorted(straight_line_dir.glob("*.jpg")):
            images.append((img, "straight_line"))
    
    # Outside-in images
    outside_in_dir = DATA_DIR / "outside_in"
    if outside_in_dir.exists():
        for img in sorted(outside_in_dir.glob("*.jpg")):
            images.append((img, "outside_in"))
    
    return images

# Get test images for parametrization
TEST_IMAGES = get_all_test_images()

@pytest.mark.parametrize("image_path,folder_type", TEST_IMAGES, 
                         ids=[f"{p.parent.name}/{p.name}" for p, _ in TEST_IMAGES])
class TestPairMatching:
    """Test suite for pair matching on all test images."""
    
    def test_sock_count(self, image_path, folder_type, predictor, resnet_model, output_dir):
        """
        Test that exactly 10 socks are detected in the image.
        
        This is a prerequisite for the pairing test - if we don't detect
        10 socks, we can't verify the pairing is correct.
        """
        resnet, preprocess, device = resnet_model
        
        # Run detection pipeline
        masks, boxes, embeddings, pairs_data, top_pairs, total_socks = run_detection_pipeline(
            image_path, predictor, resnet, preprocess, device, top_n_pairs=5
        )
        
        # Assert exactly 10 socks detected
        assert total_socks == 10, (
            f"Expected 10 socks, detected {total_socks} in {image_path.name}"
        )
    
    def test_pair_matching(self, image_path, folder_type, predictor, resnet_model, output_dir):
        """
        Test that detected pairs have correct RELATIVE ordering within the paired socks.
        
        This test only considers the 10 socks that are actually paired (5 pairs x 2 socks).
        It checks if the relative positions within those 10 socks match the expected pattern.
        
        This allows testing the pairing logic independently of detection accuracy -
        even if SAM3 detects extra socks, we can verify that the pairing algorithm
        is correctly matching the most similar socks.
        
        For straight_line: pairs should be (1,2), (3,4), (5,6), (7,8), (9,10)
        For outside_in: pairs should be (1,10), (2,9), (3,8), (4,7), (5,6)
        """
        resnet, preprocess, device = resnet_model
        
        # Run detection pipeline
        masks, boxes, embeddings, pairs_data, top_pairs, total_socks = run_detection_pipeline(
            image_path, predictor, resnet, preprocess, device, top_n_pairs=5
        )
        
        # Get expected pairs for this folder type
        expected_pairs = get_expected_pairs(folder_type)
        
        # Prepare output path
        output_subdir = output_dir / folder_type
        output_path = output_subdir / f"{image_path.stem}_result.jpg"
        
        # Handle insufficient pairs - still save visualization before skipping
        if len(top_pairs) < 5:
            # Save what we have for debugging
            if len(boxes) > 0:
                sorted_indices = sort_socks_by_y_position(boxes)
                relative_pairs = get_relative_pair_positions(top_pairs, boxes) if top_pairs else []
                create_result_visualization(
                    image_path=image_path,
                    pairs_data=pairs_data,
                    boxes=boxes,
                    sorted_indices=sorted_indices,
                    expected_pairs=expected_pairs,
                    detected_pairs=set(relative_pairs),
                    total_socks=total_socks,
                    output_path=output_path
                )
            pytest.skip(f"Only {len(top_pairs)} pairs detected, need 5. Visualization saved to: {output_path}")
        
        # Get relative positions within the paired socks only
        relative_pairs = get_relative_pair_positions(top_pairs, boxes)
        
        # Check if ordering matches expected pattern
        is_correct, explanation = check_pair_ordering(relative_pairs, folder_type)
        
        # Save visualization (using relative pairs info)
        sorted_indices = sort_socks_by_y_position(boxes)
        
        create_result_visualization(
            image_path=image_path,
            pairs_data=pairs_data,
            boxes=boxes,
            sorted_indices=sorted_indices,
            expected_pairs=expected_pairs,
            detected_pairs=set(relative_pairs),
            total_socks=total_socks,
            output_path=output_path
        )
        
        assert is_correct, (
            f"Relative pair ordering mismatch in {image_path.name}\n"
            f"  {explanation}\n"
            f"  Total socks detected: {total_socks}\n"
            f"  Result saved to: {output_path}"
        )

class TestDetectionOnly:
    """Simpler tests that only check detection, not pairing."""
    
    @pytest.mark.parametrize("image_path,folder_type", TEST_IMAGES[:1] if TEST_IMAGES else [],
                             ids=["smoke_test"])
    def test_smoke_detection(self, image_path, folder_type, predictor, resnet_model):
        """
        Smoke test: verify the pipeline runs without errors on one image.
        """
        resnet, preprocess, device = resnet_model
        
        masks, boxes, embeddings, pairs_data, top_pairs, total_socks = run_detection_pipeline(
            image_path, predictor, resnet, preprocess, device, top_n_pairs=5
        )
        
        # Just verify we got some results
        assert total_socks > 0, "No socks detected at all"
        assert len(top_pairs) > 0, "No pairs matched"

class TestExpectedPairs:
    """Unit tests for the expected pairs logic."""
    
    def test_straight_line_expected_pairs(self):
        """Test expected pairs for straight_line folder."""
        pairs = get_expected_pairs("straight_line")
        assert pairs == [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]
    
    def test_outside_in_expected_pairs(self):
        """Test expected pairs for outside_in folder."""
        pairs = get_expected_pairs("outside_in")
        assert pairs == [(1, 10), (2, 9), (3, 8), (4, 7), (5, 6)]
    
    def test_unknown_folder_raises(self):
        """Test that unknown folder type raises error."""
        with pytest.raises(ValueError):
            get_expected_pairs("unknown_folder")

class TestPositionMapping:
    """Unit tests for position mapping logic."""
    
    def test_map_detected_pairs_to_positions(self):
        """Test mapping from global indices to positions."""
        # Simulated scenario: socks are detected out of order
        # sorted_indices[0] = 5 means position 1 (topmost) has global index 5
        sorted_indices = [5, 3, 7, 1, 9, 0, 2, 8, 4, 6]
        
        # Detected pair: global indices 5 and 3 (which are positions 1 and 2)
        detected_pairs = [(5, 3)]
        
        result = map_detected_pairs_to_positions(detected_pairs, sorted_indices)
        
        assert result == {(1, 2)}
    
    def test_normalize_pairs(self):
        """Test pair normalization."""
        pairs = [(2, 1), (4, 3), (5, 6)]
        result = normalize_pairs(pairs)
        assert result == {(1, 2), (3, 4), (5, 6)}

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
