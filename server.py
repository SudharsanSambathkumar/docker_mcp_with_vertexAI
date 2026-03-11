#!/usr/bin/env python3
"""
Docker MCP Server - Model Context Protocol server for Docker management.
Built with mcp>=1.0, docker>=7.0, httpx>=0.27, anyio>=4.0
"""

import asyncio
import base64
import io
import json
import logging
import os
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import anyio
import docker
import docker.errors
from docker.models.containers import Container
from docker.models.images import Image
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    EmbeddedResource,
    ImageContent,
    TextContent,
    Tool,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("docker-mcp")

app = Server("docker-mcp-server")

# ─── Docker client (lazy) ────────────────────────────────────────────────────

_client: docker.DockerClient | None = None


def get_client() -> docker.DockerClient:
    global _client
    if _client is None:
        _client = docker.from_env()
    return _client


def _fmt_size(size_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def _container_info(c: Container) -> dict:
    c.reload()
    ports = c.ports or {}
    return {
        "id": c.short_id,
        "full_id": c.id,
        "name": c.name,
        "status": c.status,
        "image": c.image.tags[0] if c.image.tags else c.image.short_id,
        "created": c.attrs.get("Created", ""),
        "ports": {k: [p["HostPort"] for p in v] if v else [] for k, v in ports.items()},
        "labels": c.labels,
        "restart_policy": c.attrs.get("HostConfig", {}).get("RestartPolicy", {}),
        "networks": list(c.attrs.get("NetworkSettings", {}).get("Networks", {}).keys()),
    }


def _image_info(img: Image) -> dict:
    return {
        "id": img.short_id,
        "full_id": img.id,
        "tags": img.tags,
        "size": _fmt_size(img.attrs.get("Size", 0)),
        "created": img.attrs.get("Created", ""),
        "architecture": img.attrs.get("Architecture", ""),
        "os": img.attrs.get("Os", ""),
    }


# ─── Tool definitions ────────────────────────────────────────────────────────

TOOLS: list[Tool] = [
    # ── Image tools ──────────────────────────────────────────────────────────
    Tool(
        name="docker_image_list",
        description="List all local Docker images with their tags, sizes and metadata.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Filter by image name/tag (optional)"}
            },
        },
    ),
    Tool(
        name="docker_image_pull",
        description="Pull a Docker image from a registry (Docker Hub or custom).",
        inputSchema={
            "type": "object",
            "properties": {
                "image": {"type": "string", "description": "Image name and optional tag, e.g. 'nginx:latest'"},
                "registry": {"type": "string", "description": "Custom registry URL (optional)"},
                "username": {"type": "string"},
                "password": {"type": "string"},
            },
            "required": ["image"],
        },
    ),
    Tool(
        name="docker_image_push",
        description="Push a local Docker image to a registry.",
        inputSchema={
            "type": "object",
            "properties": {
                "image": {"type": "string", "description": "Image name:tag to push"},
                "username": {"type": "string"},
                "password": {"type": "string"},
                "registry": {"type": "string"},
            },
            "required": ["image"],
        },
    ),
    Tool(
        name="docker_image_build",
        description="Build a Docker image from a Dockerfile provided as text.",
        inputSchema={
            "type": "object",
            "properties": {
                "dockerfile": {"type": "string", "description": "Full Dockerfile content"},
                "tag": {"type": "string", "description": "Image tag, e.g. 'myapp:1.0'"},
                "build_args": {"type": "object", "description": "Build arguments as key-value pairs"},
                "labels": {"type": "object"},
            },
            "required": ["dockerfile", "tag"],
        },
    ),
    Tool(
        name="docker_image_remove",
        description="Remove one or more Docker images by name/tag or ID.",
        inputSchema={
            "type": "object",
            "properties": {
                "image": {"type": "string", "description": "Image name:tag or ID"},
                "force": {"type": "boolean", "default": False},
                "prune": {"type": "boolean", "default": False, "description": "Also remove dangling images"},
            },
            "required": ["image"],
        },
    ),
    Tool(
        name="docker_image_inspect",
        description="Get detailed metadata about a Docker image.",
        inputSchema={
            "type": "object",
            "properties": {
                "image": {"type": "string"}
            },
            "required": ["image"],
        },
    ),
    Tool(
        name="docker_image_tag",
        description="Tag a Docker image with a new name/tag.",
        inputSchema={
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Source image:tag"},
                "target": {"type": "string", "description": "New image:tag"},
            },
            "required": ["source", "target"],
        },
    ),
    Tool(
        name="docker_image_history",
        description="Show the build history / layers of a Docker image.",
        inputSchema={
            "type": "object",
            "properties": {
                "image": {"type": "string"}
            },
            "required": ["image"],
        },
    ),
    Tool(
        name="docker_image_prune",
        description="Remove all dangling (untagged) images. Optionally all unused images.",
        inputSchema={
            "type": "object",
            "properties": {
                "all": {"type": "boolean", "default": False, "description": "Remove ALL unused images, not just dangling"}
            },
        },
    ),
    # ── Container tools ───────────────────────────────────────────────────────
    Tool(
        name="docker_container_list",
        description="List Docker containers (running by default, optionally all).",
        inputSchema={
            "type": "object",
            "properties": {
                "all": {"type": "boolean", "default": False},
                "filters": {"type": "object", "description": "Filter dict e.g. {\"status\": \"exited\"}"},
            },
        },
    ),
    Tool(
        name="docker_container_run",
        description="Create and start a new Docker container.",
        inputSchema={
            "type": "object",
            "properties": {
                "image": {"type": "string"},
                "name": {"type": "string"},
                "command": {"type": "string"},
                "detach": {"type": "boolean", "default": True},
                "ports": {
                    "type": "object",
                    "description": "Port bindings e.g. {\"80/tcp\": 8080}"
                },
                "volumes": {
                    "type": "object",
                    "description": "Volume bindings e.g. {\"/host/path\": {\"bind\": \"/container/path\", \"mode\": \"rw\"}}"
                },
                "environment": {
                    "type": "object",
                    "description": "Environment variables as key-value pairs"
                },
                "restart_policy": {
                    "type": "string",
                    "enum": ["no", "always", "on-failure", "unless-stopped"],
                    "default": "no",
                },
                "network": {"type": "string"},
                "remove": {"type": "boolean", "default": False},
                "mem_limit": {"type": "string", "description": "Memory limit e.g. '512m'"},
                "cpu_count": {"type": "integer"},
            },
            "required": ["image"],
        },
    ),
    Tool(
        name="docker_container_stop",
        description="Stop a running container.",
        inputSchema={
            "type": "object",
            "properties": {
                "container": {"type": "string", "description": "Container name or ID"},
                "timeout": {"type": "integer", "default": 10},
            },
            "required": ["container"],
        },
    ),
    Tool(
        name="docker_container_start",
        description="Start a stopped container.",
        inputSchema={
            "type": "object",
            "properties": {
                "container": {"type": "string"}
            },
            "required": ["container"],
        },
    ),
    Tool(
        name="docker_container_restart",
        description="Restart a container.",
        inputSchema={
            "type": "object",
            "properties": {
                "container": {"type": "string"},
                "timeout": {"type": "integer", "default": 10},
            },
            "required": ["container"],
        },
    ),
    Tool(
        name="docker_container_remove",
        description="Remove a container.",
        inputSchema={
            "type": "object",
            "properties": {
                "container": {"type": "string"},
                "force": {"type": "boolean", "default": False},
                "volumes": {"type": "boolean", "default": False},
            },
            "required": ["container"],
        },
    ),
    Tool(
        name="docker_container_logs",
        description="Fetch logs from a container.",
        inputSchema={
            "type": "object",
            "properties": {
                "container": {"type": "string"},
                "tail": {"type": "integer", "default": 100},
                "timestamps": {"type": "boolean", "default": False},
                "since": {"type": "string", "description": "ISO datetime or relative e.g. '1h'"},
            },
            "required": ["container"],
        },
    ),
    Tool(
        name="docker_container_exec",
        description="Execute a command inside a running container and return output.",
        inputSchema={
            "type": "object",
            "properties": {
                "container": {"type": "string"},
                "command": {"type": "string"},
                "workdir": {"type": "string"},
                "user": {"type": "string"},
            },
            "required": ["container", "command"],
        },
    ),
    Tool(
        name="docker_container_inspect",
        description="Get detailed metadata about a container.",
        inputSchema={
            "type": "object",
            "properties": {
                "container": {"type": "string"}
            },
            "required": ["container"],
        },
    ),
    Tool(
        name="docker_container_stats",
        description="Get live resource usage stats for a container (CPU, memory, network, I/O).",
        inputSchema={
            "type": "object",
            "properties": {
                "container": {"type": "string"}
            },
            "required": ["container"],
        },
    ),
    Tool(
        name="docker_container_copy_file",
        description="Copy a text file into a running container.",
        inputSchema={
            "type": "object",
            "properties": {
                "container": {"type": "string"},
                "content": {"type": "string", "description": "File content to write"},
                "dest_path": {"type": "string", "description": "Full destination path inside the container"},
            },
            "required": ["container", "content", "dest_path"],
        },
    ),
    # ── Network tools ─────────────────────────────────────────────────────────
    Tool(
        name="docker_network_list",
        description="List all Docker networks.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="docker_network_create",
        description="Create a Docker network.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "driver": {"type": "string", "default": "bridge"},
                "labels": {"type": "object"},
                "internal": {"type": "boolean", "default": False},
            },
            "required": ["name"],
        },
    ),
    Tool(
        name="docker_network_remove",
        description="Remove a Docker network.",
        inputSchema={
            "type": "object",
            "properties": {
                "network": {"type": "string"}
            },
            "required": ["network"],
        },
    ),
    Tool(
        name="docker_network_connect",
        description="Connect a container to a network.",
        inputSchema={
            "type": "object",
            "properties": {
                "network": {"type": "string"},
                "container": {"type": "string"},
                "aliases": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["network", "container"],
        },
    ),
    # ── Volume tools ─────────────────────────────────────────────────────────
    Tool(
        name="docker_volume_list",
        description="List all Docker volumes.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="docker_volume_create",
        description="Create a Docker volume.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "driver": {"type": "string", "default": "local"},
                "labels": {"type": "object"},
            },
            "required": ["name"],
        },
    ),
    Tool(
        name="docker_volume_remove",
        description="Remove a Docker volume.",
        inputSchema={
            "type": "object",
            "properties": {
                "volume": {"type": "string"},
                "force": {"type": "boolean", "default": False},
            },
            "required": ["volume"],
        },
    ),
    # ── System tools ─────────────────────────────────────────────────────────
    Tool(
        name="docker_system_info",
        description="Get Docker daemon info (version, resources, storage driver, etc.).",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="docker_system_prune",
        description="Remove all stopped containers, dangling images, unused networks and volumes.",
        inputSchema={
            "type": "object",
            "properties": {
                "volumes": {"type": "boolean", "default": False, "description": "Also prune volumes"}
            },
        },
    ),
    Tool(
        name="docker_system_df",
        description="Show Docker disk usage breakdown.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="docker_compose_up",
        description="Run docker-compose up with a provided compose YAML string.",
        inputSchema={
            "type": "object",
            "properties": {
                "compose_yaml": {"type": "string", "description": "docker-compose.yml content"},
                "project_name": {"type": "string"},
                "detach": {"type": "boolean", "default": True},
            },
            "required": ["compose_yaml"],
        },
    ),
]


# ─── Tool handler ────────────────────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[Tool]:
    return TOOLS


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent | ImageContent | EmbeddedResource]:
    try:
        result = await anyio.to_thread.run_sync(lambda: _dispatch(name, arguments))
        return [TextContent(type="text", text=result)]
    except docker.errors.DockerException as e:
        return [TextContent(type="text", text=f"❌ Docker error: {e}")]
    except Exception as e:
        logger.exception("Tool %s failed", name)
        return [TextContent(type="text", text=f"❌ Error: {e}")]


def _dispatch(name: str, args: dict) -> str:  # noqa: C901
    client = get_client()

    # ── Images ────────────────────────────────────────────────────────────────
    if name == "docker_image_list":
        images = client.images.list(name=args.get("name"))
        if not images:
            return "No images found."
        rows = [_image_info(img) for img in images]
        return json.dumps(rows, indent=2)

    if name == "docker_image_pull":
        auth_cfg = None
        if args.get("username"):
            auth_cfg = {"username": args["username"], "password": args.get("password", "")}
        image_name = args["image"]
        if args.get("registry"):
            image_name = f"{args['registry']}/{image_name}"
        lines = []
        for line in client.api.pull(image_name, stream=True, decode=True, auth_config=auth_cfg):
            if "status" in line:
                progress = line.get("progress", "")
                lines.append(f"{line['status']} {progress}".strip())
        img = client.images.get(image_name)
        return f"✅ Pulled {image_name}\n" + "\n".join(lines[-10:]) + f"\n\nImage ID: {img.short_id}"

    if name == "docker_image_push":
        auth_cfg = None
        if args.get("username"):
            auth_cfg = {"username": args["username"], "password": args.get("password", "")}
        image_name = args["image"]
        if args.get("registry"):
            image_name = f"{args['registry']}/{image_name}"
        lines = []
        for line in client.api.push(image_name, stream=True, decode=True, auth_config=auth_cfg):
            if "error" in line:
                return f"❌ Push error: {line['error']}"
            if "status" in line:
                lines.append(f"{line['status']} {line.get('progress', '')}".strip())
        return f"✅ Pushed {image_name}\n" + "\n".join(lines[-10:])

    if name == "docker_image_build":
        dockerfile_content = args["dockerfile"]
        tag = args["tag"]
        build_args = args.get("build_args", {})
        labels = args.get("labels", {})

        # Create a tar archive with the Dockerfile in memory
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
            encoded = dockerfile_content.encode("utf-8")
            info = tarfile.TarInfo(name="Dockerfile")
            info.size = len(encoded)
            tar.addfile(info, io.BytesIO(encoded))
        tar_buffer.seek(0)

        logs = []
        image_id = None
        for chunk in client.api.build(
            fileobj=tar_buffer,
            custom_context=True,
            tag=tag,
            buildargs=build_args,
            labels=labels,
            decode=True,
        ):
            if "stream" in chunk:
                line = chunk["stream"].strip()
                if line:
                    logs.append(line)
            if "error" in chunk:
                return f"❌ Build error: {chunk['error']}\n\nLogs:\n" + "\n".join(logs)
            if "aux" in chunk:
                image_id = chunk["aux"].get("ID", "")

        return f"✅ Built image: {tag}\nImage ID: {image_id or 'unknown'}\n\nBuild output:\n" + "\n".join(logs[-20:])

    if name == "docker_image_remove":
        img = client.images.get(args["image"])
        client.images.remove(args["image"], force=args.get("force", False))
        if args.get("prune"):
            client.images.prune(filters={"dangling": True})
        return f"✅ Removed image {args['image']} (id: {img.short_id})"

    if name == "docker_image_inspect":
        img = client.images.get(args["image"])
        attrs = img.attrs
        # Show only the most useful fields
        info = {
            "id": img.id,
            "tags": img.tags,
            "created": attrs.get("Created"),
            "architecture": attrs.get("Architecture"),
            "os": attrs.get("Os"),
            "size": _fmt_size(attrs.get("Size", 0)),
            "virtual_size": _fmt_size(attrs.get("VirtualSize", 0)),
            "entrypoint": attrs.get("Config", {}).get("Entrypoint"),
            "cmd": attrs.get("Config", {}).get("Cmd"),
            "exposed_ports": list((attrs.get("Config", {}).get("ExposedPorts") or {}).keys()),
            "env": attrs.get("Config", {}).get("Env", []),
            "layers": len(attrs.get("RootFS", {}).get("Layers", [])),
        }
        return json.dumps(info, indent=2)

    if name == "docker_image_tag":
        img = client.images.get(args["source"])
        img.tag(args["target"])
        return f"✅ Tagged {args['source']} → {args['target']}"

    if name == "docker_image_history":
        img = client.images.get(args["image"])
        history = client.api.history(img.id)
        rows = []
        for h in history:
            rows.append({
                "id": h.get("Id", "")[:12],
                "created": datetime.fromtimestamp(h.get("Created", 0)).isoformat() if h.get("Created") else "",
                "created_by": h.get("CreatedBy", "")[:80],
                "size": _fmt_size(h.get("Size", 0)),
                "comment": h.get("Comment", ""),
            })
        return json.dumps(rows, indent=2)

    if name == "docker_image_prune":
        filters = {}
        if args.get("all"):
            filters["dangling"] = False
        else:
            filters["dangling"] = True
        result = client.images.prune(filters=filters)
        reclaimed = _fmt_size(result.get("SpaceReclaimed", 0))
        removed = [img["Deleted"] for img in (result.get("ImagesDeleted") or [])]
        return f"✅ Pruned {len(removed)} image(s), reclaimed {reclaimed}\n" + "\n".join(removed)

    # ── Containers ────────────────────────────────────────────────────────────
    if name == "docker_container_list":
        containers = client.containers.list(
            all=args.get("all", False),
            filters=args.get("filters"),
        )
        if not containers:
            return "No containers found."
        return json.dumps([_container_info(c) for c in containers], indent=2)

    if name == "docker_container_run":
        kwargs: dict[str, Any] = {
            "image": args["image"],
            "detach": args.get("detach", True),
            "remove": args.get("remove", False),
        }
        if args.get("name"):
            kwargs["name"] = args["name"]
        if args.get("command"):
            kwargs["command"] = args["command"]
        if args.get("ports"):
            kwargs["ports"] = args["ports"]
        if args.get("volumes"):
            kwargs["volumes"] = args["volumes"]
        if args.get("environment"):
            kwargs["environment"] = args["environment"]
        if args.get("network"):
            kwargs["network"] = args["network"]
        if args.get("mem_limit"):
            kwargs["mem_limit"] = args["mem_limit"]
        if args.get("cpu_count"):
            kwargs["cpu_count"] = args["cpu_count"]
        policy = args.get("restart_policy", "no")
        if policy != "no":
            kwargs["restart_policy"] = {"Name": policy}

        result = client.containers.run(**kwargs)
        if isinstance(result, bytes):
            return result.decode("utf-8", errors="replace")
        return f"✅ Container started\n" + json.dumps(_container_info(result), indent=2)

    if name == "docker_container_stop":
        c = client.containers.get(args["container"])
        c.stop(timeout=args.get("timeout", 10))
        return f"✅ Stopped container {c.name} ({c.short_id})"

    if name == "docker_container_start":
        c = client.containers.get(args["container"])
        c.start()
        return f"✅ Started container {c.name} ({c.short_id})"

    if name == "docker_container_restart":
        c = client.containers.get(args["container"])
        c.restart(timeout=args.get("timeout", 10))
        return f"✅ Restarted container {c.name} ({c.short_id})"

    if name == "docker_container_remove":
        c = client.containers.get(args["container"])
        name_snapshot = c.name
        id_snapshot = c.short_id
        c.remove(force=args.get("force", False), v=args.get("volumes", False))
        return f"✅ Removed container {name_snapshot} ({id_snapshot})"

    if name == "docker_container_logs":
        c = client.containers.get(args["container"])
        kwargs = {
            "tail": args.get("tail", 100),
            "timestamps": args.get("timestamps", False),
        }
        if args.get("since"):
            kwargs["since"] = args["since"]
        logs = c.logs(**kwargs)
        return logs.decode("utf-8", errors="replace") if isinstance(logs, bytes) else str(logs)

    if name == "docker_container_exec":
        c = client.containers.get(args["container"])
        kwargs = {}
        if args.get("workdir"):
            kwargs["workdir"] = args["workdir"]
        if args.get("user"):
            kwargs["user"] = args["user"]
        result = c.exec_run(args["command"], **kwargs)
        output = result.output.decode("utf-8", errors="replace") if result.output else ""
        return f"Exit code: {result.exit_code}\n\n{output}"

    if name == "docker_container_inspect":
        c = client.containers.get(args["container"])
        return json.dumps(_container_info(c), indent=2)

    if name == "docker_container_stats":
        c = client.containers.get(args["container"])
        raw = c.stats(stream=False)

        # CPU %
        cpu_delta = raw["cpu_stats"]["cpu_usage"]["total_usage"] - raw["precpu_stats"]["cpu_usage"]["total_usage"]
        sys_delta = raw["cpu_stats"].get("system_cpu_usage", 0) - raw["precpu_stats"].get("system_cpu_usage", 0)
        num_cpus = raw["cpu_stats"].get("online_cpus", 1)
        cpu_pct = (cpu_delta / sys_delta) * num_cpus * 100.0 if sys_delta > 0 else 0.0

        # Memory
        mem_usage = raw["memory_stats"].get("usage", 0)
        mem_limit = raw["memory_stats"].get("limit", 1)
        mem_pct = (mem_usage / mem_limit) * 100.0

        # Network
        net = raw.get("networks", {})
        net_rx = sum(v.get("rx_bytes", 0) for v in net.values())
        net_tx = sum(v.get("tx_bytes", 0) for v in net.values())

        # Block I/O
        bio = raw.get("blkio_stats", {}).get("io_service_bytes_recursive") or []
        bio_read = sum(b["value"] for b in bio if b.get("op") == "Read")
        bio_write = sum(b["value"] for b in bio if b.get("op") == "Write")

        stats = {
            "container": c.name,
            "cpu_percent": round(cpu_pct, 2),
            "memory_usage": _fmt_size(mem_usage),
            "memory_limit": _fmt_size(mem_limit),
            "memory_percent": round(mem_pct, 2),
            "network_rx": _fmt_size(net_rx),
            "network_tx": _fmt_size(net_tx),
            "block_read": _fmt_size(bio_read),
            "block_write": _fmt_size(bio_write),
        }
        return json.dumps(stats, indent=2)

    if name == "docker_container_copy_file":
        c = client.containers.get(args["container"])
        dest_path = args["dest_path"]
        content = args["content"].encode("utf-8")

        dest_dir = str(Path(dest_path).parent)
        filename = Path(dest_path).name

        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
            info = tarfile.TarInfo(name=filename)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
        tar_buffer.seek(0)

        c.put_archive(dest_dir, tar_buffer)
        return f"✅ Copied file to {dest_path} in container {c.name}"

    # ── Networks ─────────────────────────────────────────────────────────────
    if name == "docker_network_list":
        networks = client.networks.list()
        rows = []
        for n in networks:
            rows.append({
                "id": n.short_id,
                "name": n.name,
                "driver": n.attrs.get("Driver"),
                "scope": n.attrs.get("Scope"),
                "internal": n.attrs.get("Internal", False),
                "containers": list(n.attrs.get("Containers", {}).keys()),
            })
        return json.dumps(rows, indent=2)

    if name == "docker_network_create":
        net = client.networks.create(
            args["name"],
            driver=args.get("driver", "bridge"),
            labels=args.get("labels"),
            internal=args.get("internal", False),
        )
        return f"✅ Created network {net.name} ({net.short_id})"

    if name == "docker_network_remove":
        net = client.networks.get(args["network"])
        net.remove()
        return f"✅ Removed network {args['network']}"

    if name == "docker_network_connect":
        net = client.networks.get(args["network"])
        aliases = args.get("aliases", [])
        net.connect(args["container"], aliases=aliases if aliases else None)
        return f"✅ Connected {args['container']} to network {args['network']}"

    # ── Volumes ──────────────────────────────────────────────────────────────
    if name == "docker_volume_list":
        volumes = client.volumes.list()
        rows = []
        for v in volumes:
            rows.append({
                "name": v.name,
                "driver": v.attrs.get("Driver"),
                "mountpoint": v.attrs.get("Mountpoint"),
                "labels": v.attrs.get("Labels"),
                "created": v.attrs.get("CreatedAt"),
            })
        return json.dumps(rows, indent=2)

    if name == "docker_volume_create":
        vol = client.volumes.create(
            name=args["name"],
            driver=args.get("driver", "local"),
            labels=args.get("labels"),
        )
        return f"✅ Created volume {vol.name} (mountpoint: {vol.attrs.get('Mountpoint')})"

    if name == "docker_volume_remove":
        vol = client.volumes.get(args["volume"])
        vol.remove(force=args.get("force", False))
        return f"✅ Removed volume {args['volume']}"

    # ── System ────────────────────────────────────────────────────────────────
    if name == "docker_system_info":
        info = client.info()
        summary = {
            "docker_version": info.get("ServerVersion"),
            "api_version": client.api.api_version,
            "containers": info.get("Containers"),
            "containers_running": info.get("ContainersRunning"),
            "images": info.get("Images"),
            "storage_driver": info.get("Driver"),
            "memory": _fmt_size(info.get("MemTotal", 0)),
            "cpus": info.get("NCPU"),
            "kernel_version": info.get("KernelVersion"),
            "operating_system": info.get("OperatingSystem"),
            "architecture": info.get("Architecture"),
        }
        return json.dumps(summary, indent=2)

    if name == "docker_system_prune":
        results = client.containers.prune()
        c_count = len(results.get("ContainersDeleted") or [])
        images_r = client.images.prune()
        i_count = len(images_r.get("ImagesDeleted") or [])
        nets_r = client.networks.prune()
        n_count = len(nets_r.get("NetworksDeleted") or [])
        space = results.get("SpaceReclaimed", 0) + images_r.get("SpaceReclaimed", 0)
        out = f"✅ System prune complete\nContainers removed: {c_count}\nImages removed: {i_count}\nNetworks removed: {n_count}\nSpace reclaimed: {_fmt_size(space)}"
        if args.get("volumes"):
            vols_r = client.volumes.prune()
            v_count = len(vols_r.get("VolumesDeleted") or [])
            space += vols_r.get("SpaceReclaimed", 0)
            out += f"\nVolumes removed: {v_count}"
        return out

    if name == "docker_system_df":
        df = client.df()
        lines = ["=== Docker Disk Usage ===\n"]

        lines.append("📦 IMAGES:")
        for img in df.get("Images", []):
            tags = img.get("RepoTags") or ["<none>"]
            lines.append(f"  {tags[0]:<45} {_fmt_size(img.get('Size', 0))}")

        lines.append("\n🐋 CONTAINERS:")
        for c in df.get("Containers", []):
            lines.append(f"  {c.get('Names', ['?'])[0]:<30} {c.get('Status',''):<15} {_fmt_size(c.get('SizeRootFs', 0))}")

        lines.append("\n💾 VOLUMES:")
        for v in df.get("Volumes", []):
            lines.append(f"  {v.get('Name',''):<40} {_fmt_size(v.get('UsageData', {}).get('Size', 0))}")

        return "\n".join(lines)

    if name == "docker_compose_up":
        import subprocess
        import shutil

        compose_yaml = args["compose_yaml"]
        project_name = args.get("project_name", "mcp-compose")
        detach = args.get("detach", True)

        with tempfile.TemporaryDirectory() as tmpdir:
            compose_file = Path(tmpdir) / "docker-compose.yml"
            compose_file.write_text(compose_yaml)

            cmd = ["docker", "compose", "-p", project_name, "-f", str(compose_file), "up"]
            if detach:
                cmd.append("-d")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            output = result.stdout + result.stderr
            if result.returncode != 0:
                return f"❌ docker compose failed (exit {result.returncode}):\n{output}"
            return f"✅ docker compose up ({project_name}):\n{output}"

    return f"❌ Unknown tool: {name}"


# ─── Entry point ─────────────────────────────────────────────────────────────

async def main():
    logger.info("🐳 Docker MCP Server starting...")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    anyio.run(main)
