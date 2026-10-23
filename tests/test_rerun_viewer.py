"""Verify selectable box metadata and frame clearing without requiring a viewer."""
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rerun_viewer import log_frame


@patch("rerun_viewer.image_png", return_value=b"png")
class RerunLoggingTests(unittest.TestCase):
    def test_box_geometry_and_selection_metadata(self, decode):
        rr = Mock()
        log_frame(rr, 2, Path('photo.jpg'), [
            {'xyxy': [10, 20, 40, 70], 'class_name': 'apple',
             'class_id': 0, 'confidence': 0.875}])
        rr.set_time.assert_called_once_with('photo', sequence=2)
        rr.Clear.assert_called_once_with(recursive=True)
        rr.Boxes2D.assert_called_once_with(mins=[[10, 20]], sizes=[[30, 50]],
                                          labels=['apple 87.5%'], show_labels=True, class_ids=[0])
        self.assertEqual(rr.AnyValues.call_args.kwargs['class_name'], 'apple')
        self.assertEqual(rr.AnyValues.call_args.kwargs['confidence'], 0.875)
        self.assertEqual(rr.log.call_args.args[0], 'image/detections/box_0000')

    def test_empty_frame_clears_previous_detections(self, decode):
        rr = Mock()
        log_frame(rr, 1, Path('empty.jpg'), [])
        rr.Clear.assert_called_once_with(recursive=True)
        rr.Boxes2D.assert_not_called()
        self.assertEqual(rr.log.call_args.args[1], rr.AnyValues.return_value)


if __name__ == '__main__':
    unittest.main()
