#!/usr/bin/env python3
"""Extract CloudLab SSH targets from a manifest XML file.

Output format per node:
    [node name] [ssh hostname] [ssh username]
"""

import argparse
import sys
import xml.etree.ElementTree as ET


def local_name(tag: str) -> str:
    """Return an XML tag name without namespace."""
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def node_name(node_elem: ET.Element) -> str:
    """Pick the most useful node identifier."""
    for key in ("client_id", "component_name", "name", "component_id"):
        value = node_elem.get(key)
        if not value:
            continue
        if key == "component_id" and "+" in value:
            return value.rsplit("+", 1)[1]
        return value
    return "unknown-node"


def find_login_hostname(node_elem: ET.Element) -> str | None:
    """Find login hostname from a node's <services><login .../> block."""
    for elem in node_elem.iter():
        if local_name(elem.tag) != "login":
            continue
        hostname = elem.get("hostname")
        if hostname:
            return hostname
    return None


def parse_manifest(manifest_path: str, username: str) -> list[tuple[str, str, str]]:
    tree = ET.parse(manifest_path)
    root = tree.getroot()

    rows: list[tuple[str, str, str]] = []
    for elem in root.iter():
        if local_name(elem.tag) != "node":
            continue
        hostname = find_login_hostname(elem)
        if not hostname:
            continue
        rows.append((node_name(elem), hostname, username))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract [node name] [ssh hostname] [ssh username] from a CloudLab manifest."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to CloudLab manifest XML file.",
    )
    parser.add_argument(
        "--username",
        required=True,
        help="SSH username to print in output.",
    )
    args = parser.parse_args()

    try:
        rows = parse_manifest(args.manifest, args.username)
    except ET.ParseError as exc:
        print(f"error: invalid XML manifest: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError:
        print(f"error: manifest file not found: {args.manifest}", file=sys.stderr)
        return 2

    if not rows:
        print("error: no nodes with login hostnames found in manifest", file=sys.stderr)
        return 1

    rows.sort(key=lambda row: row[0])
    for node, hostname, username in rows:
        print(f"{node} {hostname} {username}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
