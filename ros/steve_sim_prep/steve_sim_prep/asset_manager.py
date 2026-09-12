import os
import re
from ament_index_python.packages import get_package_share_directory

class AssetManager:
    """Simple, direct asset manager resolving URDF paths for Gazebo simulation."""

    DEFAULT_CONTAINER_ROOT = "/home/ws/src/parallel_robotics_lab/third_party/shelf_gym_repo/shelf_gym/meshes/urdf/ycb_objects"

    def __init__(self, assets_dir=None, logger=None):
        self.logger = logger
        try:
            self.pkg_share = get_package_share_directory('steve_sim_prep')
        except Exception:
            self.pkg_share = '/home/ws/src/parallel_robotics_lab/ros/steve_sim_prep'

        # Resolve assets root cleanly: param -> env var -> container default
        self.ycb_root = None
        for candidate in [assets_dir, os.environ.get("YCB_ROOT"), self.DEFAULT_CONTAINER_ROOT]:
            if candidate and os.path.isdir(candidate):
                self.ycb_root = candidate
                break

    def log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(f'[AssetManager] {msg}')

    def list_available_objects(self):
        """Returns all available YCB object folder names."""
        objs = ['mustard_bottle (built-in)']
        if self.ycb_root and os.path.isdir(self.ycb_root):
            for d in sorted(os.listdir(self.ycb_root)):
                if os.path.isfile(os.path.join(self.ycb_root, d, "model.urdf")):
                    objs.append(d)
        return objs

    def compute_z_offset(self, urdf_path, urdf_content):
        """Auto-computes Z offset so object sits flush on the tabletop."""
        model_dir = os.path.dirname(urdf_path) if urdf_path else ""
        mesh_matches = re.findall(
            r'<mesh\s+[^>]*filename=["\']([^"\']+)["\'](?:\s+[^>]*scale=["\']([^"\']+)["\'])?',
            urdf_content
        )
        for fname, scale_str in mesh_matches:
            mesh_path = fname if os.path.isabs(fname) else os.path.join(model_dir, os.path.basename(fname))
            if not os.path.isfile(mesh_path):
                continue
            scale_z = 1.0
            if scale_str:
                parts = scale_str.strip().split()
                if len(parts) == 3:
                    try:
                        scale_z = float(parts[2])
                    except ValueError:
                        pass
            if mesh_path.endswith('.obj'):
                try:
                    zs = []
                    with open(mesh_path, 'r', errors='ignore') as f:
                        for line in f:
                            if line.startswith('v '):
                                parts = line.split()
                                if len(parts) >= 4:
                                    zs.append(float(parts[3]) * scale_z)
                    if zs:
                        min_z = min(zs)
                        offset = -min_z if min_z < 0 else 0.0
                        self.log(f"Auto-computed Z-offset for {os.path.basename(model_dir)}: min_z={min_z:.4f}m -> z_offset={offset:.4f}m")
                        return offset
                except Exception as e:
                    self.log(f"Warning: could not parse OBJ vertices: {e}")

        # Fallback for primitive box or cylinder
        box_match = re.search(r'<box\s+[^>]*size=["\']([^"\']+)["\']', urdf_content)
        if box_match:
            try:
                parts = [float(p) for p in box_match.group(1).replace(',', ' ').split() if p]
                if len(parts) >= 3:
                    return parts[2] / 2.0
            except Exception:
                pass

        cyl_match = re.search(r'<cylinder\s+[^>]*length=["\']([^"\']+)["\']', urdf_content)
        if cyl_match:
            try:
                return float(cyl_match.group(1).strip()) / 2.0
            except Exception:
                pass

        return 0.05

    def fix_mesh_paths_and_physics(self, urdf_content, base_dir, entity_name):
        """Remaps mesh relative filenames to absolute paths and ensures Gazebo surface friction."""
        def replace_mesh(m):
            raw_uri = m.group(2)
            fname = os.path.basename(raw_uri)
            candidate = os.path.join(base_dir, fname)
            if os.path.exists(candidate):
                return f'{m.group(1)}="{candidate}"'
            return m.group(0)

        fixed = re.sub(r'(filename)=["\']([^"\']+)["\']', replace_mesh, urdf_content)
        fixed = re.sub(r'<robot\s+name=["\'][^"\']*["\']', f'<robot name="{entity_name}"', fixed, count=1)
        if '<gazebo reference=' not in fixed:
            friction_xml = f"""
  <gazebo reference="baseLink">
    <mu1>1.0</mu1>
    <mu2>1.0</mu2>
    <selfCollide>false</selfCollide>
  </gazebo>
</robot>"""
            fixed = fixed.replace('</robot>', friction_xml)
        return fixed

    def resolve_object(self, object_name):
        name = object_name.strip()

        # 1. Direct file path
        if os.path.isfile(name):
            urdf_path = os.path.abspath(name)
            entity_name = os.path.splitext(os.path.basename(urdf_path))[0]
            with open(urdf_path, 'r') as f:
                content = f.read()
            content = self.fix_mesh_paths_and_physics(content, os.path.dirname(urdf_path), entity_name)
            z_offset = self.compute_z_offset(urdf_path, content)
            return (entity_name, content, z_offset)

        # 2. Built-in mustard bottle
        if name.lower() in ['mustard', 'mustard_bottle', 'ycbmustardbottle']:
            urdf_path = os.path.join(self.pkg_share, 'models', 'ycb_objects', 'mustard_bottle', 'model.urdf')
            if os.path.isfile(urdf_path):
                model_dir = os.path.dirname(urdf_path)
                with open(urdf_path, 'r') as f:
                    content = f.read()
                content = content.replace('package://steve_gazebo_inspection/models/ycb_objects/mustard_bottle', model_dir)
                content = content.replace('package://steve_sim_prep/models/ycb_objects/mustard_bottle', model_dir)
                z_offset = self.compute_z_offset(urdf_path, content)
                return ('YcbMustardBottle', content, z_offset)

        # 3. YCB Root directory lookup (handles canonical names, aliases, and substrings)
        if self.ycb_root and os.path.isdir(self.ycb_root):
            # Direct folder match: <ycb_root>/<name>/model.urdf
            candidate = os.path.join(self.ycb_root, name, 'model.urdf')
            if os.path.isfile(candidate):
                with open(candidate, 'r') as f:
                    content = f.read()
                content = self.fix_mesh_paths_and_physics(content, os.path.dirname(candidate), name)
                z_offset = self.compute_z_offset(candidate, content)
                return (name, content, z_offset)

            # Normalized alias & substring resolution (e.g. 'chips' -> 'YcbChipsCan')
            norm_q = name.lower().replace('_', '').replace('-', '')
            norm_q = norm_q.replace('sugar', 'suger').replace('coffee', 'masterchef').replace('spam', 'pottedmeat').replace('jello', 'gelatin')
            norm_core = norm_q[3:] if norm_q.startswith('ycb') else norm_q

            available_folders = [
                d for d in os.listdir(self.ycb_root)
                if os.path.isfile(os.path.join(self.ycb_root, d, 'model.urdf'))
            ]

            # 1. Exact core match (e.g. 'chipscan' == 'chipscan')
            matched_folder = None
            for d in available_folders:
                norm_d = d.lower().replace('_', '').replace('-', '')
                norm_d_core = norm_d[3:] if norm_d.startswith('ycb') else norm_d
                if norm_core == norm_d_core or norm_q == norm_d:
                    matched_folder = d
                    break

            # 2. Substring match (e.g. 'soup' in 'tomatosoupcan')
            if not matched_folder:
                for d in available_folders:
                    norm_d = d.lower().replace('_', '').replace('-', '')
                    norm_d_core = norm_d[3:] if norm_d.startswith('ycb') else norm_d
                    if norm_core in norm_d_core or norm_d_core in norm_core:
                        matched_folder = d
                        break

            if matched_folder:
                urdf_path = os.path.join(self.ycb_root, matched_folder, 'model.urdf')
                with open(urdf_path, 'r') as f:
                    content = f.read()
                content = self.fix_mesh_paths_and_physics(content, os.path.dirname(urdf_path), matched_folder)
                z_offset = self.compute_z_offset(urdf_path, content)
                return (matched_folder, content, z_offset)

        # Fallback
        self.log(f'Object "{name}" not found. Falling back to default YcbMustardBottle.')
        return self.resolve_object('mustard_bottle')
