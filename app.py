from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString, Point
from shapely.ops import unary_union
import pulp
import triangle as tr
import traceback
import random

app = Flask(__name__)
CORS(app)

# --- 유틸리티 함수 ---
def is_visible(p1, p2, polygon_space: Polygon | MultiPolygon):
    """점 p1과 p2 사이의 시야가 polygon_space에 의해 막히지 않는지 확인"""
    line = LineString([p1, p2])
    if polygon_space.geom_type == 'MultiPolygon':
        return any(p.covers(line) for p in polygon_space.geoms)
    return polygon_space.covers(line)

def random_simple_polygon(n_vertices: int, cx=0, cy=0, avg_radius=150, irregularity=0.5):
    """더 자연스러운 형태의 무작위 단순 다각형 생성 함수"""
    if irregularity < 0: irregularity = 0
    if irregularity > 1: irregularity = 1

    angles = np.sort(np.random.uniform(0, 2 * np.pi, n_vertices))
    radii = np.random.normal(avg_radius, avg_radius * irregularity, n_vertices)
    radii = np.clip(radii, avg_radius * 0.2, avg_radius * 2)
    
    points = [(cx + r * np.cos(a), cy + r * np.sin(a)) for r, a in zip(radii, angles)]
    return Polygon(points).buffer(0)

# --- API 엔드포인트 ---
@app.route('/generate_random', methods=['GET'])
def generate_random():
    """프론트엔드 테스트를 위한 랜덤 복합 도형 생성"""
    try:
        exterior = random_simple_polygon(random.randint(8, 15), 400, 300, 250, 0.6)
        interiors = []
        for _ in range(random.randint(1, 3)):
            min_x, min_y, max_x, max_y = exterior.bounds
            attempts = 0
            while attempts < 50:
                cx, cy = random.uniform(min_x, max_x), random.uniform(min_y, max_y)
                hole = random_simple_polygon(random.randint(4, 7), cx, cy, 50, 0.4)
                if exterior.contains(hole):
                    interiors.append(hole)
                    break
                attempts += 1
        
        exterior_coords = [list(exterior.exterior.coords)]
        interior_coords = [list(i.exterior.coords) for i in interiors]

        return jsonify({"exteriors": exterior_coords, "interiors": interior_coords})

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Random generation failed."}), 500

@app.route('/calculate', methods=['POST'])
def calculate_guards():
    try:
        data = request.get_json()
        exterior_paths = data.get('exteriors', [])
        interior_paths = data.get('interiors', [])

        if not exterior_paths:
            return jsonify({"error": "At least one exterior space is required."}), 400

        # --- exterior union ---
        exterior_polygons = []
        for p in exterior_paths:
            if len(p) < 3:
                return jsonify({"error": "Each exterior must have at least 3 points."}), 400
            exterior_polygons.append(Polygon(p).buffer(0))
        total_exterior = unary_union(exterior_polygons)

        # --- interior union ---
        interior_polygons = []
        for p in interior_paths:
            if len(p) < 3:
                continue
            interior_polygons.append(Polygon(p).buffer(0))
        total_interior = unary_union(interior_polygons) if interior_polygons else None

        # final_space = exterior - interior
        if total_interior:
            final_space = total_exterior.difference(total_interior)
        else:
            final_space = total_exterior
        final_space = final_space.buffer(0)

        if final_space.is_empty:
            return jsonify({"error": "The final shape is empty."}), 400

        # triangulate input 준비
        all_vertices = []
        all_segments = []
        hole_points = []

        def process_ring(coords):
            """polygon의 한 ring을 vertices+segments로 추가"""
            nonlocal all_vertices, all_segments
            path = list(coords)[:-1] if coords[0] == coords[-1] else list(coords)
            if len(path) < 3:
                return
            start_idx = len(all_vertices)
            all_vertices.extend(path)
            for i in range(len(path)):
                all_segments.append([start_idx + i, start_idx + (i + 1) % len(path)])

        # --- 외곽 처리 ---
        for poly in (list(total_exterior.geoms) if total_exterior.geom_type == "MultiPolygon" else [total_exterior]):
            process_ring(poly.exterior.coords)

        # --- 기둥 처리 (segment + hole point) ---
        for p in interior_paths:
            if len(p) < 3:
                continue
            hole_poly = Polygon(p).buffer(0)
            if not total_exterior.contains(hole_poly):
                continue
            # 기둥 경계도 triangulation에 포함
            process_ring(hole_poly.exterior.coords)
            # triangulation용 hole representative point
            rep = hole_poly.representative_point()
            hole_points.append((rep.x, rep.y))

        if not all_vertices or not all_segments:
            return jsonify({"error": "Invalid polygon: no vertices or segments found."}), 400

        polygon_data = {
            "vertices": np.array(all_vertices, dtype=float),
            "segments": np.array(all_segments, dtype=int),
        }
        if hole_points:
            polygon_data["holes"] = np.array(hole_points, dtype=float)

        # --- triangulate ---
        try:
            triangulation = tr.triangulate(polygon_data, "p")
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": f"Triangulation threw exception: {str(e)}"}), 500

        if not triangulation or "triangles" not in triangulation or "vertices" not in triangulation:
            keys = list(triangulation.keys()) if triangulation else []
            return jsonify({
                "error": "Triangulation failed or returned incomplete result.",
                "triangulation_keys": keys,
                "triangle_input_summary": {
                    "n_vertices": len(all_vertices),
                    "n_segments": len(all_segments),
                    "n_holes": len(hole_points),
                }
            }), 400

        # --- 삼각형 조각 만들기 ---
        tri_vertices = np.array(triangulation["vertices"], dtype=float)
        tri_idx = np.array(triangulation["triangles"], dtype=int)
        final_pieces = []
        for tri in tri_idx:
            try:
                coords = tri_vertices[tri]
                final_pieces.append(Polygon(coords))
            except Exception:
                continue

        if not final_pieces:
            return jsonify({"error": "No valid triangle pieces produced."}), 400

        # --- guard 후보: 외곽 vertex들 ---
        guard_candidates = all_vertices
        n_candidates = len(guard_candidates)
        num_pieces = len(final_pieces)

        V = np.zeros((n_candidates, num_pieces), dtype=int)
        for i, guard_pos in enumerate(guard_candidates):
            for k, piece in enumerate(final_pieces):
                try:
                    if all(is_visible(guard_pos, v, final_space) for v in piece.exterior.coords[:-1]):
                        V[i, k] = 1
                except Exception:
                    V[i, k] = 0

        # set cover 문제 설정
        x = pulp.LpVariable.dicts("x", range(n_candidates), cat="Binary")
        prob = pulp.LpProblem("SetCover", pulp.LpMinimize)
        prob += pulp.lpSum([x[i] for i in range(n_candidates)])

        for k in range(num_pieces):
            cand = [x[i] for i in range(n_candidates) if V[i, k] == 1]
            if cand:
                prob += pulp.lpSum(cand) >= 1
            else:
                return jsonify({"error": f"No guard candidate can see triangle {k}."}), 400

        try:
            solver = pulp.PULP_CBC_CMD(msg=0)
            prob.solve(solver)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": f"Solver failed: {str(e)}"}), 500

        guard_indices = [i for i in range(n_candidates) if pulp.value(x[i]) is not None and pulp.value(x[i]) > 0.5]
        
        guard_details = []
        for g_idx in guard_indices:
            guard_pos = guard_candidates[g_idx]
            visible_pieces = [p for k, p in enumerate(final_pieces) if V[g_idx, k] == 1]
            vision_area = unary_union(visible_pieces) if visible_pieces else None

            vision_coords = []
            if vision_area:
                geoms = list(vision_area.geoms) if vision_area.geom_type == "MultiPolygon" else [vision_area]
                for geom in geoms:
                    if geom.geom_type == "Polygon":
                        vision_coords.append(list(geom.exterior.coords))

            guard_details.append({"position": guard_pos, "vision_area": vision_coords})

        return jsonify({
            "guards": guard_details,
            "final_space": {
                "exteriors": [list(poly.exterior.coords) for poly in (list(final_space.geoms) if final_space.geom_type == "MultiPolygon" else [final_space])],
                "interiors": [list(h.exterior.coords) for h in interior_polygons] if interior_polygons else []
            }
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Calculation failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
