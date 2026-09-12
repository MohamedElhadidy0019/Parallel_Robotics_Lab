import os
import subprocess
import re
from ament_index_python.packages import get_package_share_directory

class AssetManager:
    """Resolves and fetches inspection target objects and environments on the fly."""
    
    DEFAULT_SHELF_GYM_URL = 'https://github.com/NilsDengler/manipulation_enhanced_map_prediction.git'
    
    def __init__(self, logger=None):
        self.logger = logger
        try:
            self.pkg_share = get_package_share_directory('steve_sim_prep')
        except Exception:
            self.pkg_share = '/home/ws/src/steve_sim_prep'
        self.cache_dir = os.path.expanduser('~/.ros/models')
        os.makedirs(self.cache_dir, exist_ok=True)

    def log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(f'[AssetManager] {msg}')

    def get_shelf_gym_path(self):
        candidate_paths = [
            '/home/ws/src/third_party/shelf_gym_repo',
            '/home/ws/src/shelf_gym',
            os.path.join(self.cache_dir, 'shelf_gym'),
            '/home/djyjyh/dev/Parallel_Robotics_Lab/third_party/shelf_gym_repo'
        ]
        for p in candidate_paths:
            if os.path.isdir(p):
                return p
        return None

    def ensure_shelf_gym(self):
        existing = self.get_shelf_gym_path()
        if existing:
            self.log(f'Using existing shelf_gym at: {existing}')
            return existing
        dest = os.path.join(self.cache_dir, 'shelf_gym')
        self.log(f'Cloning shelf_gym repository to {dest}...')
        try:
            subprocess.run(['git', 'clone', '--depth', '1', self.DEFAULT_SHELF_GYM_URL, dest], check=True)
            return dest
        except Exception as e:
            self.log(f'Failed to clone shelf_gym: {e}')
            return None

    def fix_mesh_paths(self, urdf_content, base_dir):
        lines = []
        for line in urdf_content.splitlines():
            if "filename=" in line:
                for ext in [".obj", ".stl", ".dae"]:
                    if ext in line:
                        start = line.find("filename=") + 10
                        quote = line[start-1]
                        end = line.find(quote, start)
                        if end != -1:
                            raw_uri = line[start:end]
                            fname = os.path.basename(raw_uri)
                            candidate = os.path.join(base_dir, fname)
                            if os.path.exists(candidate):
                                line = line[:start] + candidate + line[end:]
            lines.append(line)
        return "\n".join(lines)

    def resolve_object(self, object_name):
        obj_lower = object_name.lower().strip()
        
        # 1. Built-in Mustard Bottle
        if obj_lower in ['mustard', 'mustard_bottle', 'ycbmustardbottle']:
            urdf_path = os.path.join(self.pkg_share, 'models', 'ycb_objects', 'mustard_bottle', 'model.urdf')
            model_dir = os.path.dirname(urdf_path)
            with open(urdf_path, 'r') as f:
                content = f.read()
            content = content.replace('package://steve_gazebo_inspection/models/ycb_objects/mustard_bottle', model_dir)
            content = content.replace('package://steve_sim_prep/models/ycb_objects/mustard_bottle', model_dir)
            return ('YcbMustardBottle', content, 0.45)

        # 2. Shelf Gym Assets
        shelf_gym_dir = self.ensure_shelf_gym()
        if shelf_gym_dir:
            ycb_dir = os.path.join(shelf_gym_dir, 'shelf_gym', 'meshes', 'urdf', 'ycb_objects')
            if os.path.isdir(ycb_dir):
                for root, dirs, files in os.walk(ycb_dir):
                    if 'model.urdf' in files:
                        dir_name = os.path.basename(root).lower()
                        if obj_lower in dir_name or dir_name in obj_lower:
                            urdf_path = os.path.join(root, 'model.urdf')
                            self.log(f'Found match in shelf_gym: {urdf_path}')
                            with open(urdf_path, 'r') as f:
                                content = f.read()
                            content = self.fix_mesh_paths(content, root)
                            entity_name = os.path.basename(root)
                            return (entity_name, content, 0.45)

        # 3. Direct File Path
        if os.path.isfile(object_name):
            with open(object_name, 'r') as f:
                content = f.read()
            entity_name = os.path.splitext(os.path.basename(object_name))[0]
            return (entity_name, content, 0.45)

        # Fallback to Mustard Bottle
        self.log(f'Object "{object_name}" not found, falling back to default Mustard Bottle.')
        return self.resolve_object('mustard_bottle')
