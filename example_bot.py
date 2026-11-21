"""Soban Tech - Spatial-Hashed 3D Prism AI, Laser Hold-Lock, Supply Drop & Kiting (v6)."""
import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Set, Tuple
import mechfight as mf


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def norm2(x: float, y: float) -> Tuple[float, float, float]:
    d = math.hypot(x, y)
    if d < 1e-6:
        return 0.0, 0.0, 0.0
    return x / d, y / d, d


def angle_diff(a: float, b: float) -> float:
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


def point_in_poly(x: float, y: float, ring: List[float]) -> bool:
    n = len(ring) // 2
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = ring[2 * i], ring[2 * i + 1]
        xj, yj = ring[2 * j], ring[2 * j + 1]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


PrismTuple = Tuple[float, float, float, float, float, float, List[float], List[List[float]]]


class SpatialWorld:
    """Spatial-hashed 3D prism geometry for <0.5ms wall checks, rooftop walking & bullet LOS."""

    def __init__(self, map_dir: str):
        self.min_x = -510.0
        self.max_x = 552.0
        self.min_y = -330.0
        self.max_y = 385.0
        self.cell_size = 64.0
        self.buckets: Dict[Tuple[int, int], List[PrismTuple]] = {}
        self.has_map = False

        world_path = os.path.join(map_dir, "v2", "world.json")
        if not os.path.exists(world_path):
            return
        try:
            with open(world_path, "r", encoding="utf-8") as f:
                world = json.load(f)
            b = world.get("bounds", {})
            self.min_x = float(b.get("minX", -512.0)) + 3.0
            self.max_x = float(b.get("maxX", 555.0)) - 3.0
            self.min_y = float(b.get("minY", -333.0)) + 3.0
            self.max_y = float(b.get("maxY", 388.0)) - 3.0

            for p in world.get("prisms", []):
                ring = p.get("ring", [])
                if len(ring) < 6:
                    continue
                xs = ring[0::2]
                ys = ring[1::2]
                minx, maxx = min(xs), max(xs)
                miny, maxy = min(ys), max(ys)
                pt: PrismTuple = (
                    minx,
                    maxx,
                    miny,
                    maxy,
                    float(p.get("base", -10.0)),
                    float(p.get("top", 50.0)),
                    ring,
                    p.get("holes", []),
                )
                gx0 = int(math.floor((minx - 4.0) / self.cell_size))
                gx1 = int(math.floor((maxx + 4.0) / self.cell_size))
                gy0 = int(math.floor((miny - 4.0) / self.cell_size))
                gy1 = int(math.floor((maxy + 4.0) / self.cell_size))
                for gx in range(gx0, gx1 + 1):
                    for gy in range(gy0, gy1 + 1):
                        self.buckets.setdefault((gx, gy), []).append(pt)
            self.has_map = True
        except Exception:
            pass

    def _query(self, wx: float, wy: float) -> List[PrismTuple]:
        return self.buckets.get((int(math.floor(wx / self.cell_size)), int(math.floor(wy / self.cell_size))), [])

    def is_blocked_for_mech(self, wx: float, wy: float, wz: float) -> bool:
        if wx < self.min_x or wx > self.max_x or wy < self.min_y or wy > self.max_y:
            return True
        if not self.has_map:
            return False
        cand = self._query(wx, wy)
        if not cand:
            return False

        # Check mech center + 4 perimeter points (radius ~2.3m) so body never clips walls
        pts = ((wx, wy), (wx + 2.3, wy), (wx - 2.3, wy), (wx, wy + 2.3), (wx, wy - 2.3))
        for minx, maxx, miny, maxy, base, top, ring, holes in cand:
            if wx < minx - 2.5 or wx > maxx + 2.5 or wy < miny - 2.5 or wy > maxy + 2.5:
                continue
            # Feet already on top of this roof or step_up <= 1.1m -> walkable!
            if wz >= top - 1.1:
                continue
            # High bridge above 9.0m mech head -> walkable underneath!
            if base >= wz + 9.0:
                continue
            for px, py in pts:
                if point_in_poly(px, py, ring):
                    if not any(point_in_poly(px, py, h) for h in holes):
                        return True
        return False

    def has_shot_los(self, x0: float, y0: float, z0: float, x1: float, y1: float, z1: float) -> bool:
        if not self.has_map:
            return True
        dist = math.hypot(x1 - x0, y1 - y0)
        if dist < 4.5:
            return True
        steps = max(2, int(dist / 4.0))
        for s in range(1, steps):
            t = s / steps
            px = x0 + (x1 - x0) * t
            py = y0 + (y1 - y0) * t
            pz = z0 + (z1 - z0) * t
            for minx, maxx, miny, maxy, base, top, ring, holes in self._query(px, py):
                if px < minx or px > maxx or py < miny or py > maxy or pz < base or pz > top:
                    continue
                if point_in_poly(px, py, ring):
                    if not any(point_in_poly(px, py, h) for h in holes):
                        return False
        return True

    def steer_clear(self, mx: float, my: float, mz: float, dx: float, dy: float) -> Tuple[float, float, bool]:
        d_len = math.hypot(dx, dy)
        if d_len < 1e-5:
            return 0.0, 0.0, False
        ux, uy = dx / d_len, dy / d_len
        base_ang = math.atan2(uy, ux)

        if not self.is_blocked_for_mech(mx + ux * 5.0, my + uy * 5.0, mz) and not self.is_blocked_for_mech(
            mx + ux * 10.0, my + uy * 10.0, mz
        ):
            return ux, uy, False

        for offset_deg in (25, -25, 50, -50, 80, -80, 110, -110, 145, -145, 175):
            ang = base_ang + math.radians(offset_deg)
            tx, ty = math.cos(ang), math.sin(ang)
            if not self.is_blocked_for_mech(mx + tx * 5.5, my + ty * 5.5, mz) and not self.is_blocked_for_mech(
                mx + tx * 10.5, my + ty * 10.5, mz
            ):
                return tx, ty, False

        return ux, uy, True


class Bot:
    name = "SobanTech-v6"

    def __init__(self, hello: mf.Hello):
        self.slot = int(hello.get("slot", 0))
        self.seed = int(hello.get("seed", 42))
        self.rng = random.Random(self.seed + self.slot * 997)
        cfg = hello.get("config", {})
        self.pickup_r = float(cfg.get("loot", {}).get("pickup_radius", 4.0))
        map_dir = hello.get("map", {}).get("dir", "/arena")
        self.geom = SpatialWorld(map_dir)

        self.orbit_angle = self.rng.uniform(-math.pi, math.pi)
        self.orbit_dir = 1.0 if self.rng.random() < 0.5 else -1.0
        self.strafe_dir = 1.0 if self.rng.random() < 0.5 else -1.0
        self.last_pos: Optional[List[float]] = None
        self.stuck_ticks = 0
        self.jet_ticks = 0
        self.last_threat_yaw: Optional[float] = None
        self.threat_timer = 0

    def _pick_upgrade(self, offer: Dict[str, Any]) -> int:
        choices = offer.get("choices", [])
        if not choices:
            return 0
        weights = {
            "armor": 100,
            "regen": 97,
            "hp": 93,
            "health": 93,
            "speed": 89,
            "energy": 85,
            "laser": 83,
            "lightning": 81,
            "gun_rate": 76,
            "gun_dmg": 76,
            "mortar": 71,
            "sensors": 67,
        }
        best_i = 0
        best_score = -1
        for i, c in enumerate(choices):
            cid = str(c.get("id", "")).lower()
            desc = str(c.get("desc", "")).lower()
            score = 10
            for k, w in weights.items():
                if k in cid or k in desc:
                    score = max(score, w)
            if score > best_score:
                best_score = score
                best_i = i
        return best_i

    def _weapon_priority(self, kind: str) -> int:
        return {"laser": 4, "lightning": 3, "mortar": 2, "gun": 1}.get(kind, 0)

    def _compute_aim(self, me: Dict[str, Any], target: Dict[str, Any], weapon: Optional[Dict[str, Any]]) -> List[float]:
        mx, my, mz = me["pos"]
        tx, ty, tz = target["pos"]
        vx, vy, vz = target.get("vel", [0.0, 0.0, 0.0])

        muzzle_h = float(me.get("stats", {}).get("muzzle_height", 7.0))
        kind = target.get("kind", "mech")
        if kind == "crate":
            target_center_z = tz + 1.5
        elif kind == "npc":
            target_center_z = tz + 1.8
        else:
            target_center_z = tz + 4.5
        start_z = mz + muzzle_h

        dx, dy = tx - mx, ty - my
        dist_xy = math.hypot(dx, dy)
        wkind = weapon["kind"] if weapon else "gun"
        wstats = weapon.get("stats", {}) if weapon else {}

        if wkind in ("laser", "lightning"):
            yaw = math.atan2(dy, dx)
            pitch = math.atan2(target_center_z - start_z, max(dist_xy, 0.1))
            return [yaw, clamp(pitch, -1.3, 1.3)]

        if wkind == "mortar":
            speed = float(wstats.get("speed", 72.0))
            g = float(wstats.get("gravity", 22.0))
            t_flight = dist_xy / max(speed * 0.85, 1.0)
            px = tx + vx * t_flight
            py = ty + vy * t_flight
            pz = tz + max(0.0, vz * t_flight * 0.5)
            ddx, ddy = px - mx, py - my
            r = math.hypot(ddx, ddy)
            dz = pz - (mz + 9.5)
            yaw = math.atan2(ddy, ddx)
            v2 = speed * speed
            disc = v2 * v2 - g * (g * r * r + 2.0 * dz * v2)
            if disc > 0 and r > 1.0:
                pitch = math.atan((v2 - math.sqrt(disc)) / (g * r))
            else:
                pitch = 0.65
            return [yaw, clamp(pitch, -1.2, 1.35)]

        speed = float(wstats.get("speed", 252.0))
        t_flight = dist_xy / max(speed, 1.0)
        px = tx + vx * t_flight
        py = ty + vy * t_flight
        pz = target_center_z + vz * t_flight * 0.5
        yaw = math.atan2(py - my, px - mx)
        pitch = math.atan2(pz - start_z, max(math.hypot(px - mx, py - my), 0.1))
        return [yaw, clamp(pitch, -1.3, 1.3)]

    def _dodge_vector(self, me: Dict[str, Any], projectiles: List[Dict[str, Any]]) -> Tuple[float, float]:
        mx, my, _ = me["pos"]
        dodge_x, dodge_y = 0.0, 0.0
        for p in projectiles:
            px, py, _ = p["pos"]
            vx, vy, _ = p["vel"]
            v_xy = math.hypot(vx, vy)
            if v_xy < 5.0:
                continue
            rx, ry = mx - px, my - py
            dist = math.hypot(rx, ry)
            if dist > 95.0 or dist < 0.5:
                continue
            ux, uy = vx / v_xy, vy / v_xy
            proj = rx * ux + ry * uy
            if proj < 0:
                continue
            perp_x = rx - proj * ux
            perp_y = ry - proj * uy
            perp_d = math.hypot(perp_x, perp_y)
            if perp_d < 8.5:
                if perp_d > 0.1:
                    sx, sy = perp_x / perp_d, perp_y / perp_d
                else:
                    sx, sy = -uy * self.strafe_dir, ux * self.strafe_dir
                urgency = (8.5 - perp_d) / 8.5
                dodge_x += sx * urgency
                dodge_y += sy * urgency
        return dodge_x, dodge_y

    def act(self, obs: mf.Observation) -> mf.Action:
        tick = int(obs["tick"])
        t_sec = float(obs["t"])
        me = obs["self"]
        mx, my, mz = me["pos"]
        muzzle_z = mz + float(me.get("stats", {}).get("muzzle_height", 7.0))
        hp = float(me["hp"])
        max_hp = float(me["max_hp"])
        hp_ratio = hp / max(max_hp, 1.0)
        energy = float(me["energy"])
        cubes = int(me.get("cubes", 0))
        in_zone = bool(me["in_zone"])
        weapons = me.get("weapons", [None, None])
        act: mf.Action = {"tick": tick}

        # 1. Resolve upgrade offers immediately
        if obs.get("offer"):
            act["pick"] = self._pick_upgrade(obs["offer"])

        # 2. Track damage & beam threats to snap front armor
        took_damage = False
        for ev in obs.get("events", []):
            if ev.get("type") == "damaged":
                took_damage = True
                if ev.get("bearing") is not None:
                    self.last_threat_yaw = float(ev["bearing"])
                    self.threat_timer = 25

        for b in obs.get("beams", []):
            path = b.get("path", [])
            if path:
                bx, by = float(path[0][0]), float(path[0][1])
                self.last_threat_yaw = math.atan2(by - my, bx - mx)
                self.threat_timer = max(self.threat_timer, 15)

        if self.threat_timer > 0:
            self.threat_timer -= 1
        else:
            self.last_threat_yaw = None

        # 3. Wall stuck recovery
        if self.last_pos is not None:
            moved = math.hypot(mx - self.last_pos[0], my - self.last_pos[1])
            if moved < 0.22:
                self.stuck_ticks += 1
            else:
                self.stuck_ticks = max(0, self.stuck_ticks - 1)
        self.last_pos = [mx, my, mz]

        if self.stuck_ticks >= 5:
            self.jet_ticks = 9
            self.orbit_dir *= -1.0
            self.strafe_dir *= -1.0
            self.orbit_angle += 0.85 * self.orbit_dir
            self.stuck_ticks = 0

        # 4. Weapon Loot Management (Laser > Lightning > Mortar > Gun)
        pickups = obs.get("pickups", [])
        held_kinds = {w["kind"] for w in weapons if w is not None}
        best_pickup = None
        best_pickup_prio = 0
        for p in pickups:
            pkind = p["kind"]
            if pkind in held_kinds:
                continue
            d_xy = math.hypot(p["pos"][0] - mx, p["pos"][1] - my)
            d_z = abs(p["pos"][2] - mz)
            if d_xy <= self.pickup_r - 0.1 and d_z <= 3.8:
                prio = self._weapon_priority(pkind)
                if prio > best_pickup_prio:
                    best_pickup_prio = prio
                    best_pickup = p

        if best_pickup is not None:
            if weapons[1] is None:
                act["take"] = best_pickup["id"]
            elif weapons[0] is None:
                act["take"] = best_pickup["id"]
            else:
                p0 = self._weapon_priority(weapons[0]["kind"])
                p1 = self._weapon_priority(weapons[1]["kind"])
                w0_charging = weapons[0]["kind"] == "laser" and weapons[0].get("charge", 0) > 0
                w1_charging = weapons[1]["kind"] == "laser" and weapons[1].get("charge", 0) > 0
                if p0 <= p1 and best_pickup_prio > p0 and not w0_charging:
                    act["take"] = best_pickup["id"]
                    act["drop"] = 0
                elif p1 < p0 and best_pickup_prio > p1 and not w1_charging:
                    act["take"] = best_pickup["id"]
                    act["drop"] = 1

        # 5. Zone Orbit Waypoint
        zone = obs["zone"]
        ncx, ncy = float(zone["next_center"][0]), float(zone["next_center"][1])
        nrad = float(zone["next_radius"])
        ccx, ccy = float(zone["center"][0]), float(zone["center"][1])
        crad = float(zone["radius"])

        self.orbit_angle += 0.013 * self.orbit_dir
        orbit_r = max(4.0, nrad * 0.75 - 6.0)
        goal_x = ncx + math.cos(self.orbit_angle) * orbit_r
        goal_y = ncy + math.sin(self.orbit_angle) * orbit_r

        dist_curr_center = math.hypot(mx - ccx, my - ccy)
        dist_next_center = math.hypot(mx - ncx, my - ncy)
        urgent_zone = (not in_zone) or (dist_curr_center > crad - 16.0) or (
            zone["shrink_start"] - t_sec < 10.0 and dist_next_center > nrad - 8.0
        )

        # 6. Entities & Sound Radar
        all_actors = obs.get("mechs", [])
        enemies = sorted(
            [m for m in all_actors if m.get("kind") == "mech"],
            key=lambda m: math.hypot(m["pos"][0] - mx, m["pos"][1] - my),
        )
        npcs = sorted(
            [m for m in all_actors if m.get("kind") == "npc"],
            key=lambda m: math.hypot(m["pos"][0] - mx, m["pos"][1] - my),
        )
        crates = sorted(
            obs.get("crates", []),
            key=lambda c: math.hypot(c["pos"][0] - mx, c["pos"][1] - my),
        )
        visible_cubes = sorted(
            obs.get("cubes", []),
            key=lambda c: math.hypot(c["pos"][0] - mx, c["pos"][1] - my),
        )
        near_sounds = [s for s in obs.get("sounds", []) if s.get("dist") == "near"]

        # 7. Movement Logic
        move_x, move_y = 0.0, 0.0
        boost = False
        jump = False

        if urgent_zone:
            move_x, move_y, _ = norm2(goal_x - mx, goal_y - my)
            boost = energy > 22.0
        elif enemies:
            nearest = enemies[0]
            ex, ey = float(nearest["pos"][0]), float(nearest["pos"][1])
            to_ex, to_ey, edist = norm2(ex - mx, ey - my)

            if tick % 18 == 0 or nearest.get("charging"):
                self.strafe_dir *= -1.0

            strafe_x = -to_ey * self.strafe_dir
            strafe_y = to_ex * self.strafe_dir

            # 1v1 aggressive kite vs multi-enemy / low-HP disengage for 45 HP/s regen
            if len(enemies) == 1 and hp_ratio > 0.56 and cubes >= int(nearest.get("cubes", 0)):
                has_lightning = any(w and w["kind"] == "lightning" for w in weapons)
                ideal_r = 36.0 if has_lightning else 70.0
                radial = 0.45 if edist > ideal_r + 12.0 else (-0.55 if edist < ideal_r - 10.0 else 0.0)
            else:
                radial = -0.90

            zx, zy, _ = norm2(goal_x - mx, goal_y - my)
            z_w = 0.55 if dist_next_center > nrad - 14.0 else 0.25
            move_x = strafe_x * 0.92 + to_ex * radial + zx * z_w
            move_y = strafe_y * 0.92 + to_ey * radial + zy * z_w

            if (len(enemies) >= 2 or hp_ratio < 0.50) and me.get("grounded", False) and energy > 45.0 and tick % 35 == 0:
                jump = True
                self.jet_ticks = 7
            boost = (edist < 32.0 or nearest.get("charging", False) or len(enemies) >= 2 or hp_ratio < 0.52) and energy > 25.0
        else:
            nav_x, nav_y = goal_x, goal_y
            # Supply drop capture (3 cubes + FULL HP + FULL ENERGY!)
            captured_drop = False
            for d in obs.get("drops", []):
                dd = math.hypot(d["pos"][0] - mx, d["pos"][1] - my)
                if dd < 85.0 and d.get("lands_in", 99) <= 10.0 and d.get("holder") in (None, me["id"]):
                    if math.hypot(d["pos"][0] - ncx, d["pos"][1] - ncy) < nrad - 8.0:
                        nav_x, nav_y = float(d["pos"][0]), float(d["pos"][1])
                        captured_drop = True
                        break

            if not captured_drop:
                if near_sounds and hp_ratio < 0.72:
                    sb = float(near_sounds[0]["bearing"])
                    nav_x = mx - math.cos(sb) * 42.0 + (ncx - mx) * 0.35
                    nav_y = my - math.sin(sb) * 42.0 + (ncy - my) * 0.35
                elif visible_cubes and math.hypot(visible_cubes[0]["pos"][0] - mx, visible_cubes[0]["pos"][1] - my) < 50.0:
                    if math.hypot(visible_cubes[0]["pos"][0] - ncx, visible_cubes[0]["pos"][1] - ncy) < nrad - 6.0:
                        nav_x, nav_y = float(visible_cubes[0]["pos"][0]), float(visible_cubes[0]["pos"][1])
                elif pickups:
                    for p in pickups:
                        if p["kind"] not in held_kinds and self._weapon_priority(p["kind"]) >= 2:
                            if math.hypot(p["pos"][0] - mx, p["pos"][1] - my) < 50.0:
                                nav_x, nav_y = float(p["pos"][0]), float(p["pos"][1])
                                break
                elif t_sec < 95.0 and cubes < 9 and crates:
                    if math.hypot(crates[0]["pos"][0] - mx, crates[0]["pos"][1] - my) < 50.0:
                        nav_x, nav_y = float(crates[0]["pos"][0]), float(crates[0]["pos"][1])

            move_x, move_y, _ = norm2(nav_x - mx, nav_y - my)
            boost = (took_damage or captured_drop) and energy > 28.0

        # Projectile dodge
        dx_dodge, dy_dodge = self._dodge_vector(me, obs.get("projectiles", []))
        if dx_dodge != 0.0 or dy_dodge != 0.0:
            move_x += dx_dodge * 1.5
            move_y += dy_dodge * 1.5
            if energy > 28.0:
                boost = True

        # Spatial 3D Prism Whisker Steering
        if math.hypot(move_x, move_y) > 0.05:
            sx, sy, need_jump = self.geom.steer_clear(mx, my, mz, move_x, move_y)
            move_x, move_y = sx, sy
            if need_jump and self.jet_ticks == 0:
                self.jet_ticks = 6

        if self.jet_ticks > 0:
            self.jet_ticks -= 1
            if me.get("grounded", False):
                jump = True
            elif energy > 15.0:
                boost = True

        act["move"] = [round(move_x, 3), round(move_y, 3)]
        act["boost"] = boost
        act["jump"] = jump

        # 8. Target Selection & LOS-Verified Weapons
        primary_target: Optional[Dict[str, Any]] = None
        target_has_los = False

        if enemies:
            los_enemies = [
                e for e in enemies
                if self.geom.has_shot_los(mx, my, muzzle_z, float(e["pos"][0]), float(e["pos"][1]), float(e["pos"][2]) + 4.5)
            ]
            if los_enemies:
                primary_target = min(
                    los_enemies,
                    key=lambda m: float(m.get("hp", 1000)) * 0.45 + math.hypot(m["pos"][0] - mx, m["pos"][1] - my) * 0.55,
                )
                target_has_los = True
            else:
                primary_target = enemies[0]
                target_has_los = False
        elif npcs and hp_ratio > 0.55 and cubes < 10 and math.hypot(npcs[0]["pos"][0] - mx, npcs[0]["pos"][1] - my) < 60.0:
            if self.geom.has_shot_los(mx, my, muzzle_z, float(npcs[0]["pos"][0]), float(npcs[0]["pos"][1]), float(npcs[0]["pos"][2]) + 1.8):
                primary_target = npcs[0]
                target_has_los = True
        elif crates and t_sec < 100.0 and cubes < 9 and math.hypot(crates[0]["pos"][0] - mx, crates[0]["pos"][1] - my) < 42.0:
            if self.geom.has_shot_los(mx, my, muzzle_z, float(crates[0]["pos"][0]), float(crates[0]["pos"][1]), float(crates[0]["pos"][2]) + 1.5):
                c_target = dict(crates[0])
                c_target["kind"] = "crate"
                primary_target = c_target
                target_has_los = True

        fire = [False, False]
        reload_gun = False

        if primary_target is not None:
            tdist = math.hypot(primary_target["pos"][0] - mx, primary_target["pos"][1] - my)
            is_crate = primary_target.get("kind") == "crate"
            aim_weapon = (
                weapons[1]
                if (weapons[1] is not None and weapons[1]["kind"] in ("laser", "mortar") and not is_crate)
                else weapons[0]
            )
            aim = self._compute_aim(me, primary_target, aim_weapon)
            act["aim"] = [round(aim[0], 4), round(aim[1], 4)]

            yaw_err = abs(angle_diff(aim[0], float(me["aim"][0])))
            for slot_idx in (0, 1):
                w = weapons[slot_idx]
                if w is None:
                    continue
                wk = w["kind"]
                cd = float(w.get("cooldown", 0.0))

                if wk == "laser":
                    if is_crate or not target_has_los:
                        continue
                    charge = float(w.get("charge", 0.0))
                    max_charge = float(w.get("stats", {}).get("charge", 0.75))
                    if charge >= max_charge - 0.04:
                        # Hold trigger until yaw aligns, then release to fire!
                        fire[slot_idx] = yaw_err >= 0.18
                    elif cd <= 0.0 and tdist <= 275.0:
                        fire[slot_idx] = True
                elif wk == "lightning":
                    if not target_has_los:
                        continue
                    if cd <= 0.0 and tdist <= 44.0 and energy > 15.0 and yaw_err < 0.45:
                        fire[slot_idx] = True
                elif wk == "mortar":
                    if is_crate:
                        continue
                    if cd <= 0.0 and 18.0 <= tdist <= 220.0 and yaw_err < 0.25:
                        fire[slot_idx] = True
                elif wk == "gun":
                    ammo = int(w.get("ammo", 0))
                    rel = float(w.get("reload", 0.0))
                    if rel == 0.0 and ammo == 0:
                        reload_gun = True
                    elif target_has_los and cd <= 0.0 and rel == 0.0 and tdist <= 225.0 and yaw_err < 0.25:
                        fire[slot_idx] = True
        else:
            mid_sounds = [s for s in obs.get("sounds", []) if s.get("dist") in ("near", "mid")]
            if self.last_threat_yaw is not None:
                act["aim"] = [round(self.last_threat_yaw, 4), 0.0]
            elif mid_sounds:
                act["aim"] = [round(float(mid_sounds[0]["bearing"]), 4), 0.0]
            else:
                base_yaw = math.atan2(move_y, move_x) if math.hypot(move_x, move_y) > 0.1 else 0.0
                scan = base_yaw + math.sin(tick * 0.15) * 0.95
                act["aim"] = [round(scan, 4), 0.0]

            for w in weapons:
                if w and w["kind"] == "gun" and int(w.get("ammo", 12)) < 8 and float(w.get("reload", 0.0)) == 0.0:
                    reload_gun = True

        act["fire"] = fire
        if reload_gun:
            act["reload"] = True

        mf.draw.point([goal_x, goal_y, mz], "#40c0ff", 6)
        mf.draw.label(f"SobanTech-v6 | HP:{hp:.0f} | C:{cubes}")
        return act


mf.run(Bot)