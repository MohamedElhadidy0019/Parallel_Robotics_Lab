#!/usr/bin/env python3
"""Drop a coloured ball into Gazebo to see where a candidate pose actually is.

    python3 marker.py put start 0 0.31 1.09              base_link frame, default
    python3 marker.py put start 0 0.31 1.09 --frame world
    python3 marker.py put a 0 0.40 1.13 --color 0 1 0    a second one, green
    python3 marker.py del start
    python3 marker.py clear                              every marker this tool made

Markers are massless and collision-free, so they hang in the air and never touch the robot.
"""

import argparse
import sys
import time

import numpy as np
import rclpy
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

PREFIX = "nbv_marker_"

SDF = """<?xml version="1.0"?>
<sdf version="1.6"><model name="{name}"><static>true</static>
  <link name="link"><visual name="visual">
    <geometry><sphere><radius>{radius}</radius></sphere></geometry>
    <material><ambient>{r} {g} {b} 1</ambient><diffuse>{r} {g} {b} 1</diffuse>
      <emissive>{r} {g} {b} 1</emissive></material>
  </visual></link>
</model></sdf>"""


def to_world(node: Node, xyz, frame: str) -> np.ndarray:
    if frame == "world":
        return np.asarray(xyz, dtype=float)
    buffer = Buffer()
    TransformListener(buffer, node)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        try:
            tf = buffer.lookup_transform("world", frame, rclpy.time.Time())
            break
        except Exception:
            tf = None
    if tf is None:
        raise SystemExit(f"No TF world -> {frame}")
    from scipy.spatial.transform import Rotation

    q = tf.transform.rotation
    t = tf.transform.translation
    return Rotation.from_quat([q.x, q.y, q.z, q.w]).apply(xyz) + np.array([t.x, t.y, t.z])


def call(node: Node, client, request):
    if not client.wait_for_service(timeout_sec=10.0):
        raise SystemExit(f"Service {client.srv_name} unavailable")
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=15.0)
    return future.result()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["put", "del", "clear"])
    parser.add_argument("name", nargs="?", default="marker")
    parser.add_argument("xyz", nargs="*", type=float)
    parser.add_argument("--frame", default="base_link")
    parser.add_argument("--radius", type=float, default=0.03)
    parser.add_argument("--color", nargs=3, type=float, default=[1.0, 0.2, 0.0])
    args = parser.parse_args()

    rclpy.init()
    node = Node("nbv_marker")
    remove = node.create_client(DeleteEntity, "/delete_entity")

    if args.action == "clear":
        for index in range(32):
            call(node, remove, DeleteEntity.Request(name=f"{PREFIX}{index}"))
        print("cleared numbered markers; named ones need 'del <name>'")
    elif args.action == "del":
        result = call(node, remove, DeleteEntity.Request(name=PREFIX + args.name))
        print(f"delete {args.name}: {result.status_message if result else 'no response'}")
    else:
        if len(args.xyz) != 3:
            raise SystemExit("put needs x y z")
        entity = PREFIX + args.name
        call(node, remove, DeleteEntity.Request(name=entity))
        world = to_world(node, args.xyz, args.frame)
        r, g, b = args.color
        request = SpawnEntity.Request(
            name=entity,
            xml=SDF.format(name=entity, radius=args.radius, r=r, g=g, b=b),
        )
        request.initial_pose.position.x = float(world[0])
        request.initial_pose.position.y = float(world[1])
        request.initial_pose.position.z = float(world[2])
        result = call(node, node.create_client(SpawnEntity, "/spawn_entity"), request)
        print(f"{args.name}: {args.frame} {args.xyz} -> world "
              f"[{world[0]:.3f}, {world[1]:.3f}, {world[2]:.3f}] | "
              f"{result.status_message if result else 'no response'}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
