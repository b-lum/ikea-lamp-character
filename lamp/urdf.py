"""URDF parsing.

The URDF is the single source of truth for the robot: the viewer builds its 3D
scene from the description we parse here, and the animator enforces the joint
limits we parse here. Nothing about the robot is duplicated by hand.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Joint:
    name: str
    type: str  # "revolute" | "fixed"
    parent: str
    child: str
    origin_xyz: list[float]
    origin_rpy: list[float]
    axis: list[float] | None = None
    lower: float = 0.0
    upper: float = 0.0
    velocity: float = 0.0  # rad/s
    effort: float = 0.0


@dataclass
class Visual:
    shape: str  # "cylinder" | "sphere" | "mesh"
    origin_xyz: list[float]
    origin_rpy: list[float]
    color: list[float]  # rgba
    material: str = ""  # URDF material name; the viewer themes by this
    radius: float = 0.0
    length: float = 0.0
    mesh: str = ""


@dataclass
class Link:
    name: str
    visuals: list[Visual] = field(default_factory=list)


@dataclass
class Robot:
    name: str
    links: dict[str, Link]
    joints: dict[str, Joint]

    @property
    def movable_joints(self) -> dict[str, Joint]:
        return {n: j for n, j in self.joints.items() if j.type == "revolute"}

    def to_viewer_dict(self) -> dict:
        """JSON-serializable robot description sent to the viewer on connect."""
        return {
            "name": self.name,
            "links": {
                name: [vars(v) for v in link.visuals]
                for name, link in self.links.items()
            },
            "joints": [vars(j) for j in self.joints.values()],
        }


def _floats(s: str | None, default: str = "0 0 0") -> list[float]:
    return [float(x) for x in (s or default).split()]


def load(path: str | Path) -> Robot:
    root = ET.parse(path).getroot()

    materials = {}
    for mat in root.findall("material"):
        color = mat.find("color")
        if color is not None:
            materials[mat.get("name")] = _floats(color.get("rgba"), "1 1 1 1")

    links: dict[str, Link] = {}
    for link_el in root.findall("link"):
        link = Link(name=link_el.get("name"))
        for vis in link_el.findall("visual"):
            origin = vis.find("origin")
            xyz = _floats(origin.get("xyz")) if origin is not None else [0, 0, 0]
            rpy = _floats(origin.get("rpy")) if origin is not None else [0, 0, 0]
            mat = vis.find("material")
            mat_name = mat.get("name") if mat is not None else ""
            color = materials.get(mat_name, [1, 1, 1, 1])
            geo = vis.find("geometry")
            cyl, sph, mesh = geo.find("cylinder"), geo.find("sphere"), geo.find("mesh")
            if cyl is not None:
                link.visuals.append(Visual("cylinder", xyz, rpy, color, mat_name,
                                           radius=float(cyl.get("radius")),
                                           length=float(cyl.get("length"))))
            elif sph is not None:
                link.visuals.append(Visual("sphere", xyz, rpy, color, mat_name,
                                           radius=float(sph.get("radius"))))
            elif mesh is not None:
                link.visuals.append(Visual("mesh", xyz, rpy, color, mat_name,
                                           mesh=Path(mesh.get("filename")).name))
        links[link.name] = link

    joints: dict[str, Joint] = {}
    for j_el in root.findall("joint"):
        origin = j_el.find("origin")
        axis = j_el.find("axis")
        limit = j_el.find("limit")
        joint = Joint(
            name=j_el.get("name"),
            type=j_el.get("type"),
            parent=j_el.find("parent").get("link"),
            child=j_el.find("child").get("link"),
            origin_xyz=_floats(origin.get("xyz")) if origin is not None else [0, 0, 0],
            origin_rpy=_floats(origin.get("rpy")) if origin is not None else [0, 0, 0],
            axis=_floats(axis.get("xyz")) if axis is not None else None,
        )
        if limit is not None:
            joint.lower = float(limit.get("lower", 0))
            joint.upper = float(limit.get("upper", 0))
            joint.velocity = float(limit.get("velocity", 0))
            joint.effort = float(limit.get("effort", 0))
        joints[joint.name] = joint

    return Robot(name=root.get("name"), links=links, joints=joints)
