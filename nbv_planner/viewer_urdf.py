"""Rewrite a URDF so the Rerun viewer can load its meshes."""

import os
import tempfile
import xml.etree.ElementTree as ET

VIEWER_MESH_FORMATS = (".obj", ".stl", ".glb", ".gltf")
_CACHE_DIR = None


def _cache_dir() -> str:
    global _CACHE_DIR
    _CACHE_DIR = _CACHE_DIR or tempfile.mkdtemp(prefix="nbv_viewer_urdf_")
    return _CACHE_DIR


def resolve_mesh(filename: str, urdf_dir: str) -> str | None:
    """Absolute path for a URDF mesh reference, resolving package:// through ament when available."""
    if filename.startswith("file://"):
        filename = filename[len("file://"):]
    if filename.startswith("package://"):
        package, _, relative = filename[len("package://"):].partition("/")
        try:
            from ament_index_python.packages import get_package_share_directory

            filename = os.path.join(get_package_share_directory(package), relative)
        except Exception:
            return None
    elif not os.path.isabs(filename):
        filename = os.path.join(urdf_dir, filename)
    return filename if os.path.isfile(filename) else None


def viewer_mesh(path: str) -> str | None:
    """A mesh the viewer reads. COLLADA is XML, so it is converted to OBJ once and cached."""
    if path.lower().endswith(VIEWER_MESH_FORMATS):
        return path
    stem = os.path.splitext(path)[0]
    for extension in VIEWER_MESH_FORMATS:
        if os.path.isfile(stem + extension):
            return stem + extension

    # GLB keeps the materials that COLLADA carries; OBJ would drop them and render white.
    target = os.path.join(_cache_dir(), os.path.basename(stem) + ".glb")
    if not os.path.isfile(target):
        import trimesh

        try:
            trimesh.load(path).export(target)
        except Exception:
            return None
    return target


def viewer_urdf(urdf_xml: str, urdf_dir: str = ".", name: str = "viewer_robot.urdf") -> str:
    """Write a copy of the URDF with viewer-readable absolute mesh paths, dropping visuals that have none."""
    root = ET.fromstring(urdf_xml)
    for link in root.findall("link"):
        for visual in list(link.findall("visual")):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            resolved = resolve_mesh(mesh.get("filename", ""), urdf_dir)
            usable = viewer_mesh(resolved) if resolved else None
            if usable is None:
                link.remove(visual)
            else:
                mesh.set("filename", usable)
        for collision in list(link.findall("collision")):
            link.remove(collision)

    path = os.path.join(_cache_dir(), name)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def viewer_urdf_from_file(urdf_path: str) -> str:
    with open(urdf_path, "r") as f:
        return viewer_urdf(f.read(), os.path.dirname(os.path.abspath(urdf_path)), name=os.path.basename(urdf_path))
