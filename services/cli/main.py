"""CLI for the image annotation and retrieval system."""

import json
import sys
from typing import Optional

import click
import requests


# Default service URLs (can be overridden via environment or options)
DEFAULT_UPLOAD_URL = "http://localhost:8001"
DEFAULT_ANNOTATION_URL = "http://localhost:8002"
DEFAULT_QUERY_URL = "http://localhost:8003"


class ServiceConfig:
    """Configuration for service URLs."""

    def __init__(
        self,
        upload_url: str = DEFAULT_UPLOAD_URL,
        annotation_url: str = DEFAULT_ANNOTATION_URL,
        query_url: str = DEFAULT_QUERY_URL,
    ):
        self.upload_url = upload_url
        self.annotation_url = annotation_url
        self.query_url = query_url


pass_config = click.make_pass_decorator(ServiceConfig, ensure=True)


@click.group()
@click.option(
    "--upload-url",
    default=DEFAULT_UPLOAD_URL,
    envvar="UPLOAD_SERVICE_URL",
    help="Upload service URL",
)
@click.option(
    "--annotation-url",
    default=DEFAULT_ANNOTATION_URL,
    envvar="ANNOTATION_SERVICE_URL",
    help="Annotation service URL",
)
@click.option(
    "--query-url",
    default=DEFAULT_QUERY_URL,
    envvar="QUERY_SERVICE_URL",
    help="Query service URL",
)
@click.pass_context
def cli(ctx, upload_url: str, annotation_url: str, query_url: str):
    """Image annotation and retrieval CLI.

    Upload images, query for similar objects, and correct annotations.
    """
    ctx.obj = ServiceConfig(
        upload_url=upload_url,
        annotation_url=annotation_url,
        query_url=query_url,
    )


# ============================================================================
# Upload Command
# ============================================================================


@cli.command()
@click.argument("path")
@click.option("--source", default="cli", help="Source identifier")
@pass_config
def upload(config: ServiceConfig, path: str, source: str):
    """Upload an image to the pipeline.

    PATH is the path to the image file.
    """
    try:
        response = requests.post(
            f"{config.upload_url}/images",
            json={"path": path, "source": source},
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()
        click.echo(f"Image ID: {data['image_id']}")

        if data.get("already_exists"):
            click.echo("(Image was already uploaded)")

    except requests.exceptions.ConnectionError:
        click.echo(f"Error: Could not connect to upload service at {config.upload_url}", err=True)
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        click.echo(f"Error: {e.response.status_code} - {e.response.text}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


# ============================================================================
# Query Commands
# ============================================================================


@cli.command()
@click.option("--text", "query_text", help="Text query")
@click.option("--image", "query_image", help="Image path for visual query")
@click.option("--top-k", default=5, help="Number of results to return")
@click.option("--json-output", is_flag=True, help="Output as JSON")
@pass_config
def query(
    config: ServiceConfig,
    query_text: Optional[str],
    query_image: Optional[str],
    top_k: int,
    json_output: bool,
):
    """Query for similar objects.

    Use --text for text queries or --image for visual queries.
    """
    if not query_text and not query_image:
        click.echo("Error: Must provide either --text or --image", err=True)
        sys.exit(1)

    if query_text and query_image:
        click.echo("Error: Cannot use both --text and --image", err=True)
        sys.exit(1)

    try:
        if query_text:
            payload = {"kind": "text", "value": query_text, "top_k": top_k}
        else:
            payload = {"kind": "image", "value": query_image, "top_k": top_k}

        response = requests.post(
            f"{config.query_url}/query",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()

        if json_output:
            click.echo(json.dumps(data, indent=2))
        else:
            results = data.get("results", [])
            if not results:
                click.echo("No results found.")
            else:
                click.echo(f"Found {len(results)} results:\n")
                for i, result in enumerate(results, 1):
                    click.echo(
                        f"  {i}. {result['image_id']}/{result['object_id']}: "
                        f"{result['label']} (score: {result['score']:.3f})"
                    )

    except requests.exceptions.ConnectionError:
        click.echo(f"Error: Could not connect to query service at {config.query_url}", err=True)
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        click.echo(f"Error: {e.response.status_code} - {e.response.text}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


# ============================================================================
# Annotation Commands
# ============================================================================


@cli.command()
@click.argument("image_id")
@click.option("--json-output", is_flag=True, help="Output as JSON")
@pass_config
def get_annotation(config: ServiceConfig, image_id: str, json_output: bool):
    """Get annotation for an image.

    IMAGE_ID is the unique image identifier.
    """
    try:
        response = requests.get(
            f"{config.annotation_url}/annotations/{image_id}",
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()

        if json_output:
            click.echo(json.dumps(data, indent=2))
        else:
            click.echo(f"Image: {data['image_id']}")
            click.echo(f"Status: {data['status']}")
            click.echo(f"Model Version: {data['model_version']}")
            click.echo(f"\nObjects ({len(data['objects'])}):")
            for obj in data["objects"]:
                click.echo(
                    f"  - {obj['object_id']}: {obj['label']} "
                    f"(conf: {obj['conf']:.2f}, bbox: {obj['bbox']})"
                )

    except requests.exceptions.ConnectionError:
        click.echo(
            f"Error: Could not connect to annotation service at {config.annotation_url}",
            err=True,
        )
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            click.echo(f"Error: Annotation not found for image {image_id}", err=True)
        else:
            click.echo(f"Error: {e.response.status_code} - {e.response.text}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("image_id")
@click.argument("patch_file", type=click.Path(exists=True))
@click.option("--reviewer", default="cli_user", help="Reviewer identifier")
@pass_config
def correct(config: ServiceConfig, image_id: str, patch_file: str, reviewer: str):
    """Apply a correction to an annotation.

    IMAGE_ID is the unique image identifier.
    PATCH_FILE is a JSON file containing the correction patch.

    Example patch file:
    {"objects.0.label": "truck"}
    """
    try:
        with open(patch_file) as f:
            patch = json.load(f)

        response = requests.patch(
            f"{config.annotation_url}/annotations/{image_id}",
            json={"patch": patch, "reviewer": reviewer},
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()
        click.echo(f"Correction submitted: {data['message']}")

    except json.JSONDecodeError:
        click.echo(f"Error: Invalid JSON in patch file {patch_file}", err=True)
        sys.exit(1)
    except requests.exceptions.ConnectionError:
        click.echo(
            f"Error: Could not connect to annotation service at {config.annotation_url}",
            err=True,
        )
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            click.echo(f"Error: Annotation not found for image {image_id}", err=True)
        else:
            click.echo(f"Error: {e.response.status_code} - {e.response.text}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("image_id")
@click.argument("object_index", type=int)
@click.argument("new_label")
@click.option("--reviewer", default="cli_user", help="Reviewer identifier")
@pass_config
def relabel(
    config: ServiceConfig,
    image_id: str,
    object_index: int,
    new_label: str,
    reviewer: str,
):
    """Change the label of an object in an annotation.

    This is a convenience command for simple label corrections.

    IMAGE_ID is the unique image identifier.
    OBJECT_INDEX is the zero-based index of the object to correct.
    NEW_LABEL is the corrected label.
    """
    try:
        patch = {f"objects.{object_index}.label": new_label}

        response = requests.patch(
            f"{config.annotation_url}/annotations/{image_id}",
            json={"patch": patch, "reviewer": reviewer},
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()
        click.echo(f"Label updated: object {object_index} -> '{new_label}'")
        click.echo(f"Correction submitted: {data['message']}")

    except requests.exceptions.ConnectionError:
        click.echo(
            f"Error: Could not connect to annotation service at {config.annotation_url}",
            err=True,
        )
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            click.echo(f"Error: Annotation not found for image {image_id}", err=True)
        else:
            click.echo(f"Error: {e.response.status_code} - {e.response.text}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


# ============================================================================
# Health Check Commands
# ============================================================================


@cli.command()
@pass_config
def health(config: ServiceConfig):
    """Check health of all services."""
    services = [
        ("Upload", config.upload_url),
        ("Annotation", config.annotation_url),
        ("Query", config.query_url),
    ]

    all_healthy = True
    for name, url in services:
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                click.echo(f"  {name}: ✓ healthy")
            else:
                click.echo(f"  {name}: ✗ unhealthy (status {response.status_code})")
                all_healthy = False
        except requests.exceptions.ConnectionError:
            click.echo(f"  {name}: ✗ unreachable ({url})")
            all_healthy = False
        except Exception as e:
            click.echo(f"  {name}: ✗ error ({e})")
            all_healthy = False

    if not all_healthy:
        sys.exit(1)


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
