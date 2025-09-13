from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString, box
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
    if polygon_space.geom_type == 'MultiPolygon':
        return any(p.covers(LineString([p1, p2])) for p in polygon_space.geoms)
    return polygon_space.covers(LineString([p1, p2]))

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

        exterior_polygons = [Polygon(p) for p in exterior_paths]
        total_exterior = unary_union(exterior_polygons)
        
        interior_polygons = [Polygon(p) for p in interior_paths]
        total_interior = unary_union(interior_polygons)
        
        final_space = total_exterior.difference(total_interior)
        final_space = final_space.buffer(0)

        if final_space.is_empty:
             return jsonify({"error": "The final shape is empty."}), 400

        all_vertices = []
        all_segments = []
        hole_points = []
        
        def process_polygon(poly, is_hole=False):
            nonlocal all_vertices, all_segments, hole_points
            start_idx = len(all_vertices)
            path = list(poly.exterior.coords)[:-1]
            all_vertices.extend(path)
            path_len = len(path)
            for i in range(path_len):
                all_segments.append([start_idx + i, start_idx + (i + 1) % path_len])
            if is_hole:
                hole_points.append(poly.representative_point().coords[0])

        polygons_to_process = list(final_space.geoms) if final_space.geom_type == 'MultiPolygon' else [final_space]

        for poly in polygons_to_process:
            process_polygon(poly)
            for interior_ring in poly.interiors:
                interior_poly = Polygon(interior_ring).buffer(0)
                if len(interior_poly.exterior.coords) >= 3:
                    hole_points.append(interior_poly.representative_point().coords[0])
        
        polygon_data = dict(vertices=np.array(all_vertices), segments=np.array(all_segments))
        if hole_points:
            polygon_data['holes'] = np.array(hole_points)

        triangulation = tr.triangulate(polygon_data, 'p')
        final_pieces = [Polygon(triangulation['vertices'][tri]) for tri in triangulation['triangles']]
        num_pieces = len(final_pieces)
        
        guard_candidates = all_vertices
        n_candidates = len(guard_candidates)

        V = np.zeros((n_candidates, num_pieces), dtype=int)
        for i, guard_pos in enumerate(guard_candidates):
            for k, piece in enumerate(final_pieces):
                if all(is_visible(guard_pos, v, final_space) for v in piece.exterior.coords[:-1]):
                    V[i, k] = 1

        x = pulp.LpVariable.dicts("x", range(n_candidates), cat="Binary")
        prob = pulp.LpProblem("SetCover", pulp.LpMinimize)
        prob += pulp.lpSum([x[i] for i in range(n_candidates)])

        for k in range(num_pieces):
            candidate_guards_for_piece = [x[i] for i in range(n_candidates) if V[i, k] == 1]
            if candidate_guards_for_piece:
                prob += pulp.lpSum(candidate_guards_for_piece) >= 1

        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        guard_indices = [i for i in range(n_candidates) if pulp.value(x[i]) > 0.5]
        
        guard_details = []
        for g_idx in guard_indices:
            guard_pos = guard_candidates[g_idx]
            visible_pieces = [p for k, p in enumerate(final_pieces) if V[g_idx, k] == 1]
            vision_area = unary_union(visible_pieces)
            
            vision_coords = []
            geoms = list(vision_area.geoms) if vision_area.geom_type == 'MultiPolygon' else [vision_area]
            for geom in geoms:
                if geom.geom_type == 'Polygon':
                    vision_coords.append(list(geom.exterior.coords))

            guard_details.append({"position": guard_pos, "vision_area": vision_coords})

        final_space_coords = []
        for poly in polygons_to_process:
            final_space_coords.append({
                "exterior": list(poly.exterior.coords),
                "interiors": [list(i.coords) for i in poly.interiors]
            })

        return jsonify({"guards": guard_details, "final_space": final_space_coords})

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Calculation failed due to an internal error."}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)


