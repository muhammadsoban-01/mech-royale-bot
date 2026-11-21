"""Soban Tech - Map-Aware A* Pathfinding, Building Cover & Survival AI (v3)."""
import heapq
import math
import os
import random
import struct
import json
from typing import Any, Dict, List, Optional, Tuple
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


class ArenaMap:
    """Loads /arena/v2/world.json and terrain.i16 to build a fast 4m grid for A* & cover."""

    def __init__(self, map_dir: str):
        self.step = 4.0
        self.min_x = -516.0
        self.max_x = 560.0
        self.min_y = -336.0
        self.max_y = 392.0
        self.nx = int((self.max_x - self.min_x) / self.step) + 1
        self.ny = int((self.max_y - self.min_y) / self.step) + 1
        self.blocked = bytearray(self.nx * self.ny)
        self.cover = bytearray(self.nx * self.ny)
        self.loaded = False

        world_path = os.path.join(map_dir, "v2", "world.json")
        terr_path = os.path.join(map_dir, "v2", "terrain.i16")
        if not os.path.exists(world_path):
            return

        try:
            with open(world_path, "r", encoding="utf-8") as f:
                world = json.load(f)
        except Exception:
            return

        b = world.get("bounds", {})
        self.min_x = float(b.get("minX", -512.0)) - 4.0
        self.max_x = float(b.get("maxX", 555.0)) + 4.0
        self.min_y = float(b.get("minY", -333.0)) - 4.0
        self.max_y = float(b.get("maxY", 389.0)) + 4.0
        self.nx = int((self.max_x - self.min_x) / self.step) + 1
        self.ny = int((self.max_y - self.min_y) / self.step) + 1
        self.blocked = bytearray(self.nx * self.ny)
        self.cover = bytearray(self.nx * self.ny)

        # Load terrain heights if available
        t_info = world.get("terrain", {})
        t_nx = int(t_info.get("nx", 1070))
        t_ny = int(t_info.get("ny", 724))
        t_min_x = float(t_info.get("minX", -513.0))
        t_min_y = float(t_info.get("minY", -334.0))
        t_step = float(t_info.get("step", 1.0))
        t_scale = float(t_info.get("scale", 0.01))
        terr_data = None
        if os.path.exists(terr_path):
            try:
                with open(terr_path, "rb") as f:
                    raw = f.read()
                if len(raw) >= t_nx * t_ny * 2:
                    terr_data = struct.unpack(f"<{t_nx * t_ny}h", raw[: t_nx * t_ny * 2])
            except Exception:
                terr_data = None

        def ground_z(wx: float, wy: float) -> float:
            if terr_data is None:
                return 0.0
            ti = int(clamp((wx - t_min_x) / t_step, 0, t_nx - 1))
            tj = int(clamp((wy - t_min_y) / t_step, 0, t_ny - 1))
            return terr_data[tj * t_nx + ti] * t_scale

        # Pre-filter solid prisms (buildings, stadium, low bridges)
        solids = []
        for p in world.get("prisms", []):
            ring = p.get("ring", [])
            if len(ring) < 6:
                continue
            xs = ring[0::2]
            ys = ring[1::2]
            solids.append((
                min(xs) - 2.6,
                max(xs) + 2.6,
                min(ys) - 2.6,
                max(ys) + 2.6,
                float(p.get("base", -10.0)),
                float(p.get("top", 50.0)),
                ring,
                p.get("holes", []),
            ))

        # Rasterize blocked cells (with 2.5m mech radius padding)
        pad_offsets = [(0.0, 0.0), (2.2, 0.0), (-2.2, 0.0), (0.0, 2.2), (0.0, -2.2)]
        for gj in range(self.ny):
            wy = self.min_y + gj * self.step
            row_off = gj * self.nx
            for gi in range(self.nx):
                wx = self.min_x + gi * self.step
                if wx < self.min_x + 4.0 or wx > self.max_x - 4.0 or wy < self.min_y + 4.0 or wy > self.max_y - 4.0:
                    self.blocked[row_off + gi] = 1
                    continue
                gz = ground_z(wx, wy)
                is_wall = False
                for minx, maxx, miny, maxy, base, top, ring, holes in solids:
                    if wx < minx or wx > maxx or wy < miny or wy > maxy:
                        continue
                    # Bridge high enough to walk under (>= 9.2m clearance)
                    if base - gz >= 9.2:
                        continue
                    # Low step up (<= 1.1m) is walkable
                    if top - gz <= 1.1:
                        continue
                    for ox, oy in pad_offsets:
                        px, py = wx + ox, wy + oy
                        if point_in_poly(px, py, ring):
                            in_hole = any(point_in_poly(px, py, h) for h in holes)
                            if not in_hole:
                                is_wall = True
                                break
                    if is_wall:
                        break
                if is_wall:
                    self.blocked[row_off + gi] = 1

        # Compute cover score (number of adjacent blocked neighbors)
        for gj in range(1, self.ny - 1):
            row_off = gj * self.nx
            for gi in range(1, self.nx - 1):
                idx = row_off + gi
                if self.blocked[idx]:
                    continue
                c = (
                    self.blocked[idx - 1]
                    + self.blocked[idx + 1]
                    + self.blocked[idx - self.nx]
                    + self.blocked[idx + self.nx]
                    + self.blocked[idx - self.nx - 1]
                    + self.blocked[idx - self.nx + 1]
                    + self.blocked[idx + self.nx - 1]
                    + self.blocked[idx + self.nx + 1]
                )
                self.cover[idx] = c

        self.loaded = True

    def to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        gi = int(clamp(round((wx - self.min_x) / self.step), 0, self.nx - 1))
        gj = int(clamp(round((wy - self.min_y) / self.step), 0, self.ny - 1))
        return gi, gj

    def to_world(self, gi: int, gj: int) -> Tuple[float, float]:
        return self.min_x + gi * self.step, self.min_y + gj * self.step

    def is_blocked(self, gi: int, gj: int) -> bool:
        if gi < 0 or gi >= self.nx or gj < 0 or gj >= self.ny:
            return True
        return self.blocked[gj * self.nx + gi] != 0

    def nearest_free(self, wx: float, wy: float, max_r: int = 8) -> Tuple[int, int]:
        si, sj = self.to_grid(wx, wy)
        if not self.is_blocked(si, sj):
            return si, sj
        for r in range(1, max_r + 1):
            for di in range(-r, r + 1):
                for dj in (-r, r):
                    if not self.is_blocked(si + di, sj + dj):
                        return si + di, sj + dj
            for dj in range(-r + 1, r):
                for di in (-r, r):
                    if not self.is_blocked(si + di, sj + dj):
                        return si + di, sj + dj
        return si, sj

    def has_los(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        """Fast grid raycast: returns False if a building blocks line of sight."""
        if not self.loaded:
            return True
        dist = math.hypot(x1 - x0, y1 - y0)
        steps = int(dist / 3.0)
        if steps <= 1:
            return True
        for s in range(1, steps):
            t = s / steps
            gi, gj = self.to_grid(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
            if self.is_blocked(gi, gj):
                return False
        return True

    def find_cover_spot(
        self,
        mx: float,
        my: float,
        threats: List[Tuple[float, float]],
        zone_cx: float,
        zone_cy: float,
        zone_r: float,
        search_r: float = 44.0,
    ) -> Optional[Tuple[float, float]]:
        """Finds a walkable point inside the safe zone that breaks LOS from all threats."""
        if not self.loaded or not threats:
            return None
        si, sj = self.nearest_free(mx, my)
        rad_cells = int(search_r / self.step)
        best_pt = None
        best_score = -1e9
        safe_r2 = max(10.0, zone_r - 8.0) ** 2

        for dj in range(-rad_cells, rad_cells + 1, 2):
            gj = sj + dj
            if gj < 1 or gj >= self.ny - 1:
                continue
            for di in range(-rad_cells, rad_cells + 1, 2):
                gi = si + di
                if gi < 1 or gi >= self.nx - 1:
                    continue
                idx = gj * self.nx + gi
                if self.blocked[idx]:
                    continue
                wx, wy = self.to_world(gi, gj)
                if (wx - zone_cx) ** 2 + (wy - zone_cy) ** 2 > safe_r2:
                    continue

                hidden_from_all = True
                min_threat_d = 1e9
                for tx, ty in threats:
                    td = math.hypot(wx - tx, wy - ty)
                    if td < min_threat_d:
                        min_threat_d = td
                    if self.has_los(wx, wy, tx, ty):
                        hidden_from_all = False

                move_d = math.hypot(wx - mx, wy - my)
                cov = self.cover[idx]
                score = (120.0 if hidden_from_all else 0.0) + cov * 8.0 + min(min_threat_d, 80.0) * 0.6 - move_d * 0.8
                if score > best_score:
                    best_score = score
                    best_pt = (wx, wy)
        return best_pt

    def astar(self, sx: float, sy: float, gx: float, gy: float, max_nodes: int = 1800) -> Tuple[float, float]:
        """Returns the next immediate world waypoint toward (gx, gy) avoiding buildings."""
        if not self.loaded:
            return gx, gy
        si, sj = self.nearest_free(sx, sy)
        ti, tj = self.nearest_free(gx, gy)
        if (si, sj) == (ti, tj):
            return gx, gy
        # Direct LOS shortcut
        if self.has_los(sx, sy, gx, gy):
            return gx, gy

        open_heap: List[Tuple[float, int, int]] = []
        heapq.heappush(open_heap, (math.hypot(ti - si, tj - sj), si, sj))
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {(si, sj): 0.0}
        best_node = (si, sj)
        best_h = math.hypot(ti - si, tj - sj)

        neighbors = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414),
        ]
        expanded = 0
        while open_heap and expanded < max_nodes:
            _, ci, cj = heapq.heappop(open_heap)
            expanded += 1
            if (ci, cj) == (ti, tj):
                best_node = (ti, tj)
                break
            cur_g = g_score[(ci, cj)]
            for di, dj, cost in neighbors:
                ni, nj = ci + di, cj + dj
                if self.is_blocked(ni, nj):
                    continue
                # Prevent corner clipping through diagonal walls
                if di != 0 and dj != 0 and (self.is_blocked(ci + di, cj) or self.is_blocked(ci, cj + dj)):
                    continue
                ng = cur_g + cost
                if ng < g_score.get((ni, nj), 1e18):
                    g_score[(ni, nj)] = ng
                    h = math.hypot(ti - ni, tj - nj)
                    if h < best_h:
                        best_h = h
                        best_node = (ni, nj)
                    came_from[(ni, nj)] = (ci, cj)
                    heapq.heappush(open_heap, (ng + h * 1.15, ni, nj))

        # Reconstruct path and pick a waypoint ~3 cells ahead for smooth steering
        path = [best_node]
        cur = best_node
        while cur in came_from:
            cur = came_from[cur]
            path.append(cur)
        path.reverse()
        if len(path) <= 1:
            return gx, gy
        pick_idx = min(3, len(path) - 1)
        return self.to_world(path[pick_idx][0], path[pick_idx][1])


class Bot:
    name = "SobanTech-v3"

    def __init__(self, hello: mf.Hello):
        self.slot = int(hello.get("slot", 0))
        self.seed = int(hello.get("seed", 42))
        self.rng = random.Random(self.seed + self.slot * 997)
        cfg = hello.get("config", {})
        self.pickup_r = float(cfg.get("loot", {}).get("pickup_radius", 4.0))
        map_dir = hello.get("map", {}).get("dir", "/arena")
        self.arena = ArenaMap(map_dir)

        self.anchor_angle = self.rng.uniform(-math.pi, math.pi)
        self.strafe_dir = 1.0 if self.rng.random() < 0.5 else -1.0
        self.last_pos: Optional[List[float]] = None
        self.stuck_ticks = 0
        self.jet_ticks = 0
        self.last_threat_yaw: Optional[float] = None
        self.threat_timer = 0
        self.cached_wp: Optional[Tuple[float, float]] = None
        self.wp_tick = -99

    def _pick_upgrade(self, offer: Dict[str, Any]) -> int:
        choices = offer.get("choices", [])
        if not choices:
            return 0
        weights = {
            "armor": 100,
            "regen": 98,
            "hp": 92,
            "health": 92,
            "speed": 85,
            "energy": 80,
            "laser": 75,
            "lightning": 72,
            "mortar": 68,
            "gun_rate": 64,
            "gun_dmg": 64,
            "sensors": 60,
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

    def _pick_zone_cover_anchor(self, ncx: float, ncy: float, nrad: float) -> Tuple[float, float]:
        """Picks a cover-rich spot near the edge of the next zone."""
        edge_r = max(0.0, nrad * 0.78 - 6.0)
        gx = ncx + math.cos(self.anchor_angle) * edge_r
        gy = ncy + math.sin(self.anchor_angle) * edge_r
        if not self.arena.loaded or nrad < 15.0:
            return gx, gy

        # Search around target edge point for highest wall-cover cell
        ci, cj = self.arena.nearest_free(gx, gy, max_r=6)
        best_x, best_y = self.arena.to_world(ci, cj)
        best_score = -1e9
        max_r2 = max(8.0, nrad - 8.0) ** 2
        for dj in range(-5, 6):
            gj = cj + dj
            if gj < 1 or gj >= self.arena.ny - 1:
                continue
            for di in range(-5, 6):
                gi = ci + di
                if gi < 1 or gi >= self.arena.nx - 1:
                    continue
                idx = gj * self.arena.nx + gi
                if self.arena.blocked[idx]:
                    continue
                wx, wy = self.arena.to_world(gi, gj)
                d2 = (wx - ncx) ** 2 + (wy - ncy) ** 2
                if d2 > max_r2:
                    continue
                cov = self.arena.cover[idx]
                sc = cov * 10.0 - math.hypot(wx - gx, wy - gy) * 0.5
                if sc > best_score:
                    best_score = sc
                    best_x, best_y = wx, wy
        return best_x, best_y

    def _compute_aim(self, me: Dict[str, Any], target: Dict[str, Any], weapon: Optional[Dict[str, Any]]) -> List[float]:
        mx, my, mz = me["pos"]
        tx, ty, tz = target["pos"]
        vx, vy, vz = target.get("vel", [0.0, 0.0, 0.0])

        muzzle_h = float(me.get("stats", {}).get("muzzle_height", 7.0))
        target_center_z = tz + (1.8 if target.get("kind") == "npc" else 4.5)
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
            if perp_d < 8.0:
                if perp_d > 0.1:
                    sx, sy = perp_x / perp_d, perp_y / perp_d
                else:
                    sx, sy = -uy * self.strafe_dir, ux * self.strafe_dir
                urgency = (8.0 - perp_d) / 8.0
                dodge_x += sx * urgency
                dodge_y += sy * urgency
        return dodge_x, dodge_y

    def act(self, obs: mf.Observation) -> mf.Action:
        tick = int(obs["tick"])
        me = obs["self"]
        mx, my, mz = me["pos"]
        hp = float(me["hp"])
        max_hp = float(me["max_hp"])
        hp_ratio = hp / max(max_hp, 1.0)
        energy = float(me["energy"])
        in_zone = bool(me["in_zone"])
        weapons = me.get("weapons", [None, None])
        act: mf.Action = {"tick": tick}

        # 1. Resolve upgrade offers
        if obs.get("offer"):
            act["pick"] = self._pick_upgrade(obs["offer"])

        # 2. Track damage events (snap front armor + trigger cover retreat)
        took_damage = False
        for ev in obs.get("events", []):
            if ev.get("type") == "damaged":
                took_damage = True
                if ev.get("bearing") is not None:
                    self.last_threat_yaw = float(ev["bearing"])
                    self.threat_timer = 30

        if self.threat_timer > 0:
            self.threat_timer -= 1
        else:
            self.last_threat_yaw = None

        # 3. Wall stuck recovery
        if self.last_pos is not None:
            moved = math.hypot(mx - self.last_pos[0], my - self.last_pos[1])
            if moved < 0.16:
                self.stuck_ticks += 1
            else:
                self.stuck_ticks = max(0, self.stuck_ticks - 1)
        self.last_pos = [mx, my, mz]

        if self.stuck_ticks >= 5:
            self.jet_ticks = 8
            self.anchor_angle = (self.anchor_angle + 1.05) % (2.0 * math.pi)
            self.strafe_dir *= -1.0
            self.stuck_ticks = 0
            self.wp_tick = -99

        # 4. Weapon Loot Management
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

        # 5. Zone & Cover Anchor
        zone = obs["zone"]
        ncx, ncy = float(zone["next_center"][0]), float(zone["next_center"][1])
        nrad = float(zone["next_radius"])
        ccx, ccy = float(zone["center"][0]), float(zone["center"][1])
        crad = float(zone["radius"])

        if tick % 80 == 0 and not obs.get("mechs"):
            self.anchor_angle += 0.35 * self.strafe_dir

        goal_x, goal_y = self._pick_zone_cover_anchor(ncx, ncy, nrad)

        dist_curr_center = math.hypot(mx - ccx, my - ccy)
        dist_next_center = math.hypot(mx - ncx, my - ncy)
        urgent_zone = (not in_zone) or (dist_curr_center > crad - 16.0) or (
            zone["shrink_start"] - obs["t"] < 8.0 and dist_next_center > nrad - 10.0
        )

        # 6. Enemies & NPCs
        all_actors = obs.get("mechs", [])
        enemies = sorted(
            [m for m in all_actors if m.get("kind") == "mech"],
            key=lambda m: math.hypot(m["pos"][0] - mx, m["pos"][1] - my),
        )
        npcs = sorted(
            [m for m in all_actors if m.get("kind") == "npc"],
            key=lambda m: math.hypot(m["pos"][0] - mx, m["pos"][1] - my),
        )

        # 7. Tactical Movement (A* + Building Cover + Stealth)
        move_x, move_y = 0.0, 0.0
        boost = False
        jump = False

        if urgent_zone:
            if tick - self.wp_tick >= 4 or self.cached_wp is None:
                self.cached_wp = self.arena.astar(mx, my, goal_x, goal_y)
                self.wp_tick = tick
            wx, wy = self.cached_wp
            move_x, move_y, _ = norm2(wx - mx, wy - my)
            boost = energy > 22.0
        elif enemies:
            # Active Cover-Seeking: Duck behind a building wall to break enemy LOS & regen HP!
            threats = [(float(e["pos"][0]), float(e["pos"][1])) for e in enemies[:3]]
            cover_pt = self.arena.find_cover_spot(mx, my, threats, ncx, ncy, nrad, search_r=44.0)

            nearest = enemies[0]
            ex, ey = float(nearest["pos"][0]), float(nearest["pos"][1])
            edist = math.hypot(ex - mx, ey - my)

            # Should we retreat to cover? Almost always in 1v15 unless enemy is 1-shot!
            enemy_low = float(nearest.get("hp", 1000)) < 220.0 and len(enemies) == 1 and hp_ratio > 0.55
            if cover_pt is not None and not enemy_low:
                wp_x, wp_y = self.arena.astar(mx, my, cover_pt[0], cover_pt[1], max_nodes=700)
                cx, cy, cdist = norm2(wp_x - mx, wp_y - my)
                if cdist > 2.0:
                    move_x, move_y = cx, cy
                    boost = (took_damage or hp_ratio < 0.75 or len(enemies) >= 2) and energy > 28.0
                else:
                    # Already behind cover: hold still silently to regen 45 HP/s!
                    move_x, move_y = 0.0, 0.0
            else:
                # Fallback strafe + backpedal if in open field
                to_ex, to_ey, _ = norm2(ex - mx, ey - my)
                if tick % 24 == 0 or nearest.get("charging"):
                    self.strafe_dir *= -1.0
                strafe_x = -to_ey * self.strafe_dir
                strafe_y = to_ex * self.strafe_dir
                radial = -0.85 if (edist < 70.0 or len(enemies) >= 2) else -0.25
                zx, zy, _ = norm2(goal_x - mx, goal_y - my)
                move_x = strafe_x * 0.85 + to_ex * radial + zx * 0.35
                move_y = strafe_y * 0.85 + to_ey * radial + zy * 0.35
                boost = (edist < 35.0 or nearest.get("charging", False)) and energy > 30.0
        else:
            # Check for uncontested supply drop nearby
            target_pt = (goal_x, goal_y)
            for d in obs.get("drops", []):
                dd = math.hypot(d["pos"][0] - mx, d["pos"][1] - my)
                if dd < 65.0 and d.get("lands_in", 99) < 8.0 and d.get("holder") in (None, me["id"]):
                    if math.hypot(d["pos"][0] - ncx, d["pos"][1] - ncy) < nrad - 10.0:
                        target_pt = (float(d["pos"][0]), float(d["pos"][1]))
                        break

            # Opportunistic nearby cube
            for c in obs.get("cubes", []):
                if math.hypot(c["pos"][0] - mx, c["pos"][1] - my) < 20.0 and abs(c["pos"][2] - mz) < 4.0:
                    if math.hypot(c["pos"][0] - ncx, c["pos"][1] - ncy) < nrad - 8.0:
                        target_pt = (float(c["pos"][0]), float(c["pos"][1]))
                        break

            if tick - self.wp_tick >= 5 or self.cached_wp is None:
                self.cached_wp = self.arena.astar(mx, my, target_pt[0], target_pt[1])
                self.wp_tick = tick

            wx, wy = self.cached_wp
            gx, gy, gdist = norm2(wx - mx, wy - my)
            total_dist = math.hypot(target_pt[0] - mx, target_pt[1] - my)
            if total_dist > 5.0:
                # Stealth speed (< 5 m/s) inside safe zone so footsteps make 0 sound!
                speed_scale = 0.36 if (dist_next_center < nrad - 14.0 and not took_damage) else 1.0
                move_x, move_y = gx * speed_scale, gy * speed_scale
                boost = took_damage and energy > 30.0
            else:
                move_x, move_y = 0.0, 0.0

        # Projectile dodge
        dx_dodge, dy_dodge = self._dodge_vector(me, obs.get("projectiles", []))
        if dx_dodge != 0.0 or dy_dodge != 0.0:
            move_x += dx_dodge * 1.4
            move_y += dy_dodge * 1.4
            if energy > 35.0:
                boost = True

        m_len = math.hypot(move_x, move_y)
        if m_len > 1.0:
            move_x /= m_len
            move_y /= m_len

        if self.jet_ticks > 0:
            self.jet_ticks -= 1
            if me.get("grounded", False):
                jump = True
            elif energy > 15.0:
                boost = True

        act["move"] = [round(move_x, 3), round(move_y, 3)]
        act["boost"] = boost
        act["jump"] = jump

        # 8. Aiming & Firing (Only shoot if we have LOS so we don't reveal position through walls!)
        primary_target = None
        visible_enemies = [
            e for e in enemies
            if self.arena.has_los(mx, my, float(e["pos"][0]), float(e["pos"][1]))
        ]
        if visible_enemies:
            primary_target = min(
                visible_enemies,
                key=lambda m: math.hypot(m["pos"][0] - mx, m["pos"][1] - my) * 0.6 + float(m.get("hp", 1000)) * 0.15,
            )
        elif npcs and hp_ratio > 0.70 and dist_next_center < nrad - 18.0:
            if math.hypot(npcs[0]["pos"][0] - mx, npcs[0]["pos"][1] - my) < 48.0:
                if self.arena.has_los(mx, my, float(npcs[0]["pos"][0]), float(npcs[0]["pos"][1])):
                    primary_target = npcs[0]

        fire = [False, False]
        reload_gun = False

        if primary_target is not None:
            tdist = math.hypot(primary_target["pos"][0] - mx, primary_target["pos"][1] - my)
            aim_weapon = weapons[1] if (weapons[1] is not None and weapons[1]["kind"] in ("laser", "mortar")) else weapons[0]
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
                    charge = float(w.get("charge", 0.0))
                    max_charge = float(w.get("stats", {}).get("charge", 0.75))
                    if charge >= max_charge - 0.04 and yaw_err < 0.22:
                        fire[slot_idx] = False
                    elif cd <= 0.0 and tdist <= 275.0:
                        fire[slot_idx] = True
                elif wk == "lightning":
                    if cd <= 0.0 and tdist <= 44.0 and energy > 12.0 and yaw_err < 0.45:
                        fire[slot_idx] = True
                elif wk == "mortar":
                    if cd <= 0.0 and 16.0 <= tdist <= 220.0 and yaw_err < 0.25:
                        fire[slot_idx] = True
                elif wk == "gun":
                    ammo = int(w.get("ammo", 0))
                    rel = float(w.get("reload", 0.0))
                    if rel == 0.0 and ammo == 0:
                        reload_gun = True
                    elif cd <= 0.0 and rel == 0.0 and tdist <= 225.0 and yaw_err < 0.25:
                        fire[slot_idx] = True
        else:
            sounds = obs.get("sounds", [])
            near_sounds = [s for s in sounds if s.get("dist") in ("near", "mid")]
            if self.last_threat_yaw is not None:
                act["aim"] = [round(self.last_threat_yaw, 4), 0.0]
            elif enemies:
                # Face nearest tracked enemy behind cover so front armor protects us
                ex, ey = float(enemies[0]["pos"][0]), float(enemies[0]["pos"][1])
                act["aim"] = [round(math.atan2(ey - my, ex - mx), 4), 0.0]
            elif near_sounds:
                act["aim"] = [round(float(near_sounds[0]["bearing"]), 4), 0.0]
            else:
                base_yaw = math.atan2(move_y, move_x) if m_len > 0.1 else (self.anchor_angle + math.pi)
                scan = base_yaw + math.sin(tick * 0.12) * 0.85
                act["aim"] = [round(scan, 4), 0.0]

            for w in weapons:
                if w and w["kind"] == "gun" and int(w.get("ammo", 12)) < 6 and float(w.get("reload", 0.0)) == 0.0:
                    reload_gun = True

        act["fire"] = fire
        if reload_gun:
            act["reload"] = True

        mf.draw.point([goal_x, goal_y, mz], "#40c0ff", 6)
        mf.draw.label(f"SobanTech-v3 | HP:{hp:.0f} | E:{len(enemies)}")
        return act


mf.run(Bot)