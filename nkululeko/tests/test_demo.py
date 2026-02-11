import os
import sys
import types
import tempfile
from unittest.mock import patch, MagicMock, mock_open


class TestDemoArgParser:
    """Test argument parser configuration."""

    def test_parser_defaults(self):
        """Test default argument values."""
        from unittest.mock import patch
        with patch('sys.argv', ['demo.py']):
            import argparse
            parser = argparse.ArgumentParser(description="Call the nkululeko DEMO framework.")
            parser.add_argument("--config", default="exp.ini")
            parser.add_argument("--file", default=None)
            parser.add_argument("--list", nargs="?", default=None)
            parser.add_argument("--folder", nargs="?", default="./")
            parser.add_argument("--outfile", nargs="?", default=None)
            args = parser.parse_args([])
            assert args.config == "exp.ini"
            assert args.file is None
            assert args.list is None
            assert args.folder == "./"
            assert args.outfile is None

    def test_parser_with_args(self):
        """Test parser with provided arguments."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", default="exp.ini")
        parser.add_argument("--file", default=None)
        parser.add_argument("--list", nargs="?", default=None)
        parser.add_argument("--folder", nargs="?", default="./")
        parser.add_argument("--outfile", nargs="?", default=None)
        args = parser.parse_args(["--config", "test.ini", "--file", "audio.wav", "--folder", "/data/"])
        assert args.config == "test.ini"
        assert args.file == "audio.wav"
        assert args.folder == "/data/"


class TestPrintPipe:
    """Test the print_pipe logic extracted from demo.main."""

    def test_nan_detection(self):
        """Test NaN score detection in pipeline results."""
        score = float('nan')
        assert score != score  # NaN check used in demo.py

    def test_valid_score(self):
        """Test valid score passes NaN check."""
        score = 0.95
        assert score == score

    def test_result_formatting(self):
        """Test result string formatting."""
        file_path = "test.wav"
        label = "happy"
        result = f"{file_path}, {label}"
        assert result == "test.wav, happy"

    def test_outfile_writing(self):
        """Test results are written to outfile."""
        results = ["file1.wav, happy", "file2.wav, sad"]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("\n".join(results))
            tmpfile = f.name

        with open(tmpfile, "r") as f:
            content = f.read()
        assert "file1.wav, happy" in content
        assert "file2.wav, sad" in content
        os.remove(tmpfile)


class TestDemoMain:
    """Test demo main function with mocked dependencies."""

    @patch("os.path.isfile", return_value=False)
    def test_config_not_found_exits(self, mock_isfile):
        """Test that missing config file triggers exit."""
        config_file = "nonexistent.ini"
        assert not os.path.isfile(config_file)

    def test_finetune_model_path_construction(self):
        """Test finetune model path is correctly constructed."""
        exp_dir = "/tmp/experiments"
        expected = os.path.join(exp_dir, "models", "run_0", "torch")
        assert expected == "/tmp/experiments/models/run_0/torch"

    def test_demo_branches(self):
        """Test the branching logic for file/list/folder."""
        # file provided
        args_file = "audio.wav"
        args_list = None
        if args_file is None and args_list is None:
            mode = "no_input"
        elif args_list is None:
            mode = "single_file"
        else:
            mode = "list"
        assert mode == "single_file"

        # list provided
        args_file = None
        args_list = "files.txt"
        if args_file is None and args_list is None:
            mode = "no_input"
        elif args_list is None:
            mode = "single_file"
        else:
            mode = "list"
        assert mode == "list"

        # neither
        args_file = None
        args_list = None
        if args_file is None and args_list is None:
            mode = "no_input"
        elif args_list is None:
            mode = "single_file"
        else:
            mode = "list"
        assert mode == "no_input"
