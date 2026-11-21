"""Soban Tech - High-Survival & Tactical Combat Mech AI (v2)."""
import math
import random
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
    d = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return d


class Bot:
    name = "SobanTech-v2"

    def __init__(self, hello: mf.Hello):
        self.slot = int(hello.get("slot", 0))
        self.seed = int(hello.get("seed", 42))
        self.rng = random.Random(self.seed + self.slot * 997)
        cfg = hello.get("config", {})
        self.pickup_r = float(cfg.get("loot", {}).get("pickup_radius", 4.0))
        self.cube_r = 3.5
        self.weapons_cfg = cfg.get("weapons", {})
        self.mech_cfg = cfg.get("mech", {})

        # Each match pick a consistent quiet edge angle based on seed/slot
        self.anchor_angle = self.rng.uniform(-math.pi, math.pi)
        self.strafe_dir = 1.0 if self.rng.random() < 0.5 else -1.0
        self.last_pos: Optional[List[float]] = None
        self.stuck_ticks = 0
        self.jet_ticks = 0
        self.last_threat_yaw: Optional[float] = None
        self.threat_timer = 0
        self.scan_yaw = self.anchor_angle

    def _pick_upgrade(self, offer: Dict[str, Any]) -> int:
        choices = offer.get("choices", [])
        if not choices:
            return 0
        # Priority keywords for survival & lethality
        weights = {
            "armor": 100,
            "regen": 95,
            "hp": 90,
            "health": 90,
            "speed": 80,
            "energy": 75,
            "laser": 70,
            "lightning": 68,
            "mortar": 65,
            "gun_rate": 60,
            "gun_dmg": 60,
            "sensors": 55,
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
        target_center_z = tz + (1.8 if target.get("kind") == "npc" else 4.5)
        start_z = mz + muzzle_h

        dx, dy = tx - mx, ty - my
        dist_xy = math.hypot(dx, dy)
        wkind = weapon["kind"] if weapon else "gun"
        wstats = weapon.get("stats", {}) if weapon else {}

        if wkind == "laser" or wkind == "lightning":
            # Hitscan / instant arc
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
            # Ballistic trajectory formula: tan(theta) = (v^2 - sqrt(v^4 - g*(g*r^2 + 2*dz*v^2))) / (g*r)
            v2 = speed * speed
            disc = v2 * v2 - g * (g * r * r + 2.0 * dz * v2)
            if disc > 0 and r > 1.0:
                pitch = math.atan((v2 - math.sqrt(disc)) / (g * r))
            else:
                pitch = 0.65
            return [yaw, clamp(pitch, -1.2, 1.35)]

        # Default: Gun projectile lead
        speed = float(wstats.get("speed", 252.0))
        t_flight = dist_xy / max(speed, 1.0)
        px = tx + vx * t_flight
        py = ty + vy * t_flight
        pz = target_center_z + vz * t_flight * 0.5
        yaw = math.atan2(py - my, px - mx)
        pitch = math.atan2(pz - start_z, max(math.hypot(px - mx, py - my), 0.1))
        return [yaw, clamp(pitch, -1.3, 1.3)]

    def _dodge_vector(self, me: Dict[str, Any], projectiles: List[Dict[str, Any]]) -> Tuple[float, float]:
        mx, my, mz = me["pos"]
        dodge_x, dodge_y = 0.0, 0.0
        for p in projectiles:
            px, py, pz = p["pos"]
            vx, vy, vz = p["vel"]
            v_xy = math.hypot(vx, vy)
            if v_xy < 5.0:
                continue
            rx, ry = mx - px, my - py
            dist = math.hypot(rx, ry)
            if dist > 90.0 or dist < 0.5:
                continue
            ux, uy = vx / v_xy, vy / v_xy
            proj = rx * ux + ry * uy
            if proj < 0:
                continue  # Moving away
            # Closest perpendicular distance
            perp_x = rx - proj * ux
            perp_y = ry - proj * uy
            perp_d = math.hypot(perp_x, perp_y)
            if perp_d < 7.5:
                # Push perpendicular away from line of fire
                if perp_d > 0.1:
                    sx, sy = perp_x / perp_d, perp_y / perp_d
                else:
                    sx, sy = -uy * self.strafe_dir, ux * self.strafe_dir
                urgency = (7.5 - perp_d) / 7.5
                dodge_x += sx * urgency
                dodge_y += sy * urgency
        return dodge_x, dodge_y

    def act(self, obs: mf.Observation) -> mf.Action:
        tick = obs["tick"]
        me = obs["self"]
        mx, my, mz = me["pos"]
        hp = float(me["hp"])
        max_hp = float(me["max_hp"])
        hp_ratio = hp / max(max_hp, 1.0)
        energy = float(me["energy"])
        in_zone = bool(me["in_zone"])
        weapons = me.get("weapons", [None, None])
        act: mf.Action = {"tick": tick}

        # 1. Resolve upgrade offers immediately
        if obs.get("offer"):
            act["pick"] = self._pick_upgrade(obs["offer"])

        # 2. Track damage events to snap front armor toward attackers
        for ev in obs.get("events", []):
            if ev.get("type") == "damaged" and ev.get("bearing") is not None:
                self.last_threat_yaw = float(ev["bearing"])
                self.threat_timer = 25  # 2.5 seconds

        if self.threat_timer > 0:
            self.threat_timer -= 1
        else:
            self.last_threat_yaw = None

        # 3. Stuck detection (wall collision recovery)
        if self.last_pos is not None:
            moved = math.hypot(mx - self.last_pos[0], my - self.last_pos[1])
            if moved < 0.18:
                self.stuck_ticks += 1
            else:
                self.stuck_ticks = max(0, self.stuck_ticks - 1)
        self.last_pos = [mx, my, mz]

        if self.stuck_ticks >= 6:
            self.jet_ticks = 8
            self.anchor_angle = (self.anchor_angle + 0.9) % (2.0 * math.pi)
            self.strafe_dir *= -1.0
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
                # Replace lower-priority weapon if not charging laser
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

        # 5. Zone Edge Positioning (Stay near edge of next_center to avoid 15-bot cluster)
        zone = obs["zone"]
        ncx, ncy = zone["next_center"]
        nrad = float(zone["next_radius"])
        ccx, ccy = zone["center"]
        crad = float(zone["radius"])

        # Slowly rotate anchor angle around edge to stay unpredictable
        if tick % 30 == 0:
            self.anchor_angle += 0.18 * self.strafe_dir

        edge_dist = max(0.0, nrad * 0.76 - 6.0)
        goal_x = ncx + math.cos(self.anchor_angle) * edge_dist
        goal_y = ncy + math.sin(self.anchor_angle) * edge_dist

        # Check if currently safely inside current zone
        dist_curr_center = math.hypot(mx - ccx, my - ccy)
        dist_next_center = math.hypot(mx - ncx, my - ncy)
        urgent_zone = (not in_zone) or (dist_curr_center > crad - 18.0) or (
            zone["shrink_start"] - obs["t"] < 8.0 and dist_next_center > nrad - 10.0
        )

        # 6. Identify Enemies & NPCs
        all_actors = obs.get("mechs", [])
        enemies = sorted(
            [m for m in all_actors if m.get("kind") == "mech"],
            key=lambda m: math.hypot(m["pos"][0] - mx, m["pos"][1] - my),
        )
        npcs = sorted(
            [m for m in all_actors if m.get("kind") == "npc"],
            key=lambda m: math.hypot(m["pos"][0] - mx, m["pos"][1] - my),
        )

        # 7. Movement Vector Calculation
        move_x, move_y = 0.0, 0.0
        boost = False
        jump = False

        if urgent_zone:
            # Sprint directly into safe zone
            ux, uy, _ = norm2(goal_x - mx, goal_y - my)
            move_x, move_y = ux, uy
            boost = energy > 25.0
        elif enemies:
            # Combat kiting: maintain safe distance & strafe perpendicular to nearest enemy
            nearest = enemies[0]
            ex, ey = nearest["pos"][0], nearest["pos"][1]
            to_ex, to_ey, edist = norm2(ex - mx, ey - my)

            # Switch strafe direction periodically or if enemy charges laser
            if tick % 25 == 0 or nearest.get("charging"):
                self.strafe_dir *= -1.0

            # Perpendicular strafe vector
            strafe_x = -to_ey * self.strafe_dir
            strafe_y = to_ex * self.strafe_dir

            # Retreat if too close or low HP, otherwise keep optimal mid-range (55-85m)
            desired_range = 75.0 if hp_ratio < 0.65 else 55.0
            if edist < desired_range or len(enemies) >= 2:
                radial = -0.85  # Backpedal while facing enemy (front armor!)
            elif edist > 130.0:
                radial = 0.25
            else:
                radial = -0.25

            # Blend zone pull so we don't back into gas
            zx, zy, zdist = norm2(goal_x - mx, goal_y - my)
            z_weight = 0.55 if dist_next_center > nrad - 15.0 else 0.20

            move_x = strafe_x * 0.85 + to_ex * radial + zx * z_weight
            move_y = strafe_y * 0.85 + to_ey * radial + zy * z_weight
            boost = (edist < 35.0 or nearest.get("charging", False)) and energy > 30.0
        else:
            # No visible enemies: check opportunistic nearby cubes or move quietly to edge anchor
            cubes = obs.get("cubes", [])
            near_cube = None
            for c in cubes:
                if math.hypot(c["pos"][0] - mx, c["pos"][1] - my) < 22.0 and abs(c["pos"][2] - mz) < 4.0:
                    near_cube = c
                    break

            if near_cube is not None and dist_next_center < nrad - 8.0:
                ux, uy, _ = norm2(near_cube["pos"][0] - mx, near_cube["pos"][1] - my)
                move_x, move_y = ux, uy
            else:
                gx, gy, gdist = norm2(goal_x - mx, goal_y - my)
                if gdist > 8.0:
                    # Move silently (< 5 m/s) when already inside next_zone so enemies don't hear footsteps!
                    speed_scale = 0.38 if dist_next_center < nrad - 12.0 else 1.0
                    move_x, move_y = gx * speed_scale, gy * speed_scale
                else:
                    # Hold position silently at edge
                    move_x, move_y = 0.0, 0.0

        # Add projectile dodge impulse
        dx_dodge, dy_dodge = self._dodge_vector(me, obs.get("projectiles", []))
        if dx_dodge != 0.0 or dy_dodge != 0.0:
            move_x += dx_dodge * 1.4
            move_y += dy_dodge * 1.4
            if energy > 35.0:
                boost = True

        # Normalize move vector
        m_len = math.hypot(move_x, move_y)
        if m_len > 1.0:
            move_x /= m_len
            move_y /= m_len

        # Obstacle jump/jet recovery
        if self.jet_ticks > 0:
            self.jet_ticks -= 1
            if me.get("grounded", False):
                jump = True
            elif energy > 15.0:
                boost = True

        act["move"] = [round(move_x, 3), round(move_y, 3)]
        act["boost"] = boost
        act["jump"] = jump

        # 8. Aiming & Firing Logic
        primary_target = None
        if enemies:
            # Prioritize lowest HP enemy within reasonable range
            primary_target = min(
                enemies,
                key=lambda m: math.hypot(m["pos"][0] - mx, m["pos"][1] - my) * 0.6 + float(m.get("hp", 1000)) * 0.15,
            )
        elif npcs and hp_ratio > 0.65 and dist_next_center < nrad - 15.0:
            # Farm nearby NPC drone for 2 cubes if safe
            if math.hypot(npcs[0]["pos"][0] - mx, npcs[0]["pos"][1] - my) < 55.0:
                primary_target = npcs[0]

        fire = [False, False]
        reload_gun = False

        if primary_target is not None:
            tdist = math.hypot(primary_target["pos"][0] - mx, primary_target["pos"][1] - my)
            # Choose best weapon for aim calculation
            aim_weapon = weapons[1] if (weapons[1] is not None and weapons[1]["kind"] in ("laser", "mortar")) else weapons[0]
            aim = self._compute_aim(me, primary_target, aim_weapon)
            act["aim"] = [round(aim[0], 4), round(aim[1], 4)]

            # Check if torso is reasonably aligned before firing
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
                        # Release trigger to fire full-power beam!
                        fire[slot_idx] = False
                    elif cd <= 0.0 and tdist <= 275.0:
                        # Hold trigger to charge
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
            # No visible target: look toward recent damage/sound bearing to keep front armor up!
            sounds = obs.get("sounds", [])
            near_sounds = [s for s in sounds if s.get("dist") in ("near", "mid")]
            if self.last_threat_yaw is not None:
                act["aim"] = [round(self.last_threat_yaw, 4), 0.0]
            elif near_sounds:
                act["aim"] = [round(float(near_sounds[0]["bearing"]), 4), 0.0]
            else:
                # Gentle scan around movement direction
                if m_len > 0.1:
                    base_yaw = math.atan2(move_y, move_x)
                else:
                    base_yaw = self.anchor_angle + math.pi
                scan = base_yaw + math.sin(tick * 0.12) * 0.85
                act["aim"] = [round(scan, 4), 0.0]

            # Reload gun during downtime if low on ammo
            for w in weapons:
                if w and w["kind"] == "gun" and int(w.get("ammo", 12)) < 6 and float(w.get("reload", 0.0)) == 0.0:
                    reload_gun = True

        act["fire"] = fire
        if reload_gun:
            act["reload"] = True

        mf.draw.point([goal_x, goal_y, mz], "#40c0ff", 6)
        mf.draw.label(f"SobanTech | HP:{hp:.0f} | E:{len(enemies)}")
        return act


mf.run(Bot)