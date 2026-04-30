"""Tests for CLI commands."""

import json
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from services.cli.main import cli


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


class TestUploadCommand:
    """Tests for the upload command."""

    def test_upload_success(self, runner):
        """Should upload image and display ID."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"image_id": "img_abc123"}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response) as mock_post:
            result = runner.invoke(cli, ["upload", "/path/to/image.jpg"])

            assert result.exit_code == 0
            assert "img_abc123" in result.output
            mock_post.assert_called_once()

    def test_upload_with_source(self, runner):
        """Should pass source parameter."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"image_id": "img_test"}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response) as mock_post:
            result = runner.invoke(
                cli, ["upload", "/path/to/image.jpg", "--source", "camera_A"]
            )

            assert result.exit_code == 0
            call_kwargs = mock_post.call_args[1]
            assert call_kwargs["json"]["source"] == "camera_A"

    def test_upload_already_exists(self, runner):
        """Should indicate when image already exists."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "image_id": "img_existing",
            "already_exists": True,
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response):
            result = runner.invoke(cli, ["upload", "/path/to/image.jpg"])

            assert result.exit_code == 0
            assert "already uploaded" in result.output.lower()

    def test_upload_connection_error(self, runner):
        """Should handle connection errors gracefully."""
        import requests

        with patch("requests.post", side_effect=requests.exceptions.ConnectionError()):
            result = runner.invoke(cli, ["upload", "/path/to/image.jpg"])

            assert result.exit_code == 1
            assert "could not connect" in result.output.lower()


class TestQueryCommand:
    """Tests for the query command."""

    def test_text_query(self, runner):
        """Should execute text query."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"image_id": "img_1", "object_id": "obj_0", "label": "car", "score": 0.95},
            ],
            "query_kind": "text",
            "total_results": 1,
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response):
            result = runner.invoke(cli, ["query", "--text", "find cars"])

            assert result.exit_code == 0
            assert "car" in result.output
            assert "0.95" in result.output

    def test_image_query(self, runner):
        """Should execute image query."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"image_id": "img_1", "object_id": "obj_0", "label": "person", "score": 0.88},
            ],
            "query_kind": "image",
            "total_results": 1,
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response):
            result = runner.invoke(cli, ["query", "--image", "/path/to/query.jpg"])

            assert result.exit_code == 0
            assert "person" in result.output

    def test_query_no_results(self, runner):
        """Should handle no results."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [],
            "query_kind": "text",
            "total_results": 0,
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response):
            result = runner.invoke(cli, ["query", "--text", "nonexistent"])

            assert result.exit_code == 0
            assert "no results" in result.output.lower()

    def test_query_json_output(self, runner):
        """Should output JSON when requested."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [{"image_id": "img_1", "object_id": "obj_0", "label": "car", "score": 0.9}],
            "query_kind": "text",
            "total_results": 1,
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response):
            result = runner.invoke(cli, ["query", "--text", "car", "--json-output"])

            assert result.exit_code == 0
            # Output should be valid JSON
            data = json.loads(result.output)
            assert "results" in data

    def test_query_requires_text_or_image(self, runner):
        """Should require either --text or --image."""
        result = runner.invoke(cli, ["query"])

        assert result.exit_code == 1
        assert "must provide" in result.output.lower()

    def test_query_rejects_both_text_and_image(self, runner):
        """Should reject both --text and --image."""
        result = runner.invoke(cli, ["query", "--text", "car", "--image", "/path.jpg"])

        assert result.exit_code == 1
        assert "cannot use both" in result.output.lower()


class TestGetAnnotationCommand:
    """Tests for the get-annotation command."""

    def test_get_annotation(self, runner):
        """Should display annotation details."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "image_id": "img_test",
            "status": "pending",
            "model_version": "mock_v1",
            "objects": [
                {"object_id": "obj_0", "label": "car", "conf": 0.95, "bbox": [10, 20, 100, 200]},
            ],
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            result = runner.invoke(cli, ["get-annotation", "img_test"])

            assert result.exit_code == 0
            assert "img_test" in result.output
            assert "car" in result.output
            assert "pending" in result.output

    def test_get_annotation_not_found(self, runner):
        """Should handle not found errors."""
        import requests

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response
        )

        with patch("requests.get", return_value=mock_response):
            result = runner.invoke(cli, ["get-annotation", "nonexistent"])

            assert result.exit_code == 1
            assert "not found" in result.output.lower()


class TestCorrectCommand:
    """Tests for the correct command."""

    def test_correct_with_patch_file(self, runner):
        """Should apply correction from patch file."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "image_id": "img_test",
            "status": "submitted",
            "message": "Correction submitted by cli_user",
        }
        mock_response.raise_for_status = MagicMock()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"objects.0.label": "truck"}, f)
            patch_file = f.name

        with patch("requests.patch", return_value=mock_response):
            result = runner.invoke(cli, ["correct", "img_test", patch_file])

            assert result.exit_code == 0
            assert "submitted" in result.output.lower()

    def test_correct_with_reviewer(self, runner):
        """Should pass reviewer parameter."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": "OK"}
        mock_response.raise_for_status = MagicMock()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"objects.0.label": "truck"}, f)
            patch_file = f.name

        with patch("requests.patch", return_value=mock_response) as mock_patch:
            result = runner.invoke(
                cli, ["correct", "img_test", patch_file, "--reviewer", "alice"]
            )

            assert result.exit_code == 0
            call_kwargs = mock_patch.call_args[1]
            assert call_kwargs["json"]["reviewer"] == "alice"


class TestRelabelCommand:
    """Tests for the relabel command."""

    def test_relabel(self, runner):
        """Should change object label."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": "OK"}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.patch", return_value=mock_response) as mock_patch:
            result = runner.invoke(cli, ["relabel", "img_test", "0", "truck"])

            assert result.exit_code == 0
            assert "truck" in result.output
            call_kwargs = mock_patch.call_args[1]
            assert "objects.0.label" in call_kwargs["json"]["patch"]


class TestHealthCommand:
    """Tests for the health command."""

    def test_all_healthy(self, runner):
        """Should report all services healthy."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("requests.get", return_value=mock_response):
            result = runner.invoke(cli, ["health"])

            assert result.exit_code == 0
            assert "healthy" in result.output.lower()

    def test_service_unhealthy(self, runner):
        """Should report unhealthy services."""
        import requests

        with patch(
            "requests.get", side_effect=requests.exceptions.ConnectionError()
        ):
            result = runner.invoke(cli, ["health"])

            assert result.exit_code == 1
            assert "unreachable" in result.output.lower()


class TestServiceUrlOptions:
    """Tests for service URL configuration."""

    def test_custom_upload_url(self, runner):
        """Should use custom upload URL."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"image_id": "img_test"}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response) as mock_post:
            result = runner.invoke(
                cli,
                [
                    "--upload-url",
                    "http://custom:9000",
                    "upload",
                    "/path/to/image.jpg",
                ],
            )

            assert result.exit_code == 0
            call_args = mock_post.call_args[0]
            assert "http://custom:9000" in call_args[0]
