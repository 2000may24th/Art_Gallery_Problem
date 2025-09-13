from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import unary_union
import pulp
import triangle as tr
import traceback
import random

app = Flask(__name__)
CORS(app)

# --- 정밀도 제어를 위한 전역 설정 ---
# 모든 좌표 계산에 사용할 소수점 자리수를 정의합니다.
# 이 값을 조정하여 정밀도를 세밀하게 제어할 수 있습니다.
GEOMETRY_PRECISION = 6
# 정밀도에 기반한 허용 오차(tolerance)를 설정합니다.
# 기하학적 비교 연산(예: 점이 거의 동일한지 확인)에 사용됩니다.
TOLERANCE = 10**(-GEOMETRY_PRECISION - 1)

# --- 유틸리티 함수 (수정됨) ---
def snap_coords(coords):
    """좌표를 전역 정밀도에 맞춰 반올림(스냅)하여 부동 소수점 오류를 최소화합니다."""
    return [tuple(round(v, GEOMETRY_PRECISION) for v in p) for p in coords]

def is_visible(p1, p2, polygon_space: Polygon | MultiPolygon):
    """점 p1과 p2 사이의 시야가 polygon_space에 의해 막히지 않는지 확인 (부동 소수점 허용 오차 적용)"""
    line = LineString([p1, p2])
    
    # 두 점이 허용 오차 내에서 동일한 경우, 항상 보인다고 간주합니다.
    if line.length < TOLERANCE:
        return True

    # 공간을 아주 약간 확장(buffer)하여 선이 경계에 완벽하게 걸치는 경우를 포함시킵니다.
    # 이 허용 오차(tolerance)는 부동 소수점 오류로 인한 잘못된 '보이지 않음' 판정을 방지합니다.
    # join_style=2 (mitre), resolution=1 (빠른 처리)
    buffered_space = polygon_space.buffer(TOLERANCE, resolution=1, join_style=2)
    
    if buffered_space.geom_type == 'MultiPolygon':
        return any(p.covers(line) for p in buffered_space.geoms)
    return buffered_space.covers(line)

def random_simple_polygon(n_vertices: int, cx=0, cy=0, avg_radius=150, irregularity=0.5):
    """더 자연스러운 형태의 무작위 단순 다각형 생성 함수"""
    if irregularity < 0: irregularity = 0
    if irregularity > 1: irregularity = 1

    angles = np.sort(np.random.uniform(0, 2 * np.pi, n_vertices))
    radii = np.random.normal(avg_radius, avg_radius * irregularity, n_vertices)
    radii = np.clip(radii, avg_radius * 0.2, avg_radius * 2)
    
    points = [(cx + r * np.cos(a), cy + r * np.sin(a)) for r, a in zip(radii, angles)]
    # 생성된 좌표도 스냅핑하여 일관성을 유지합니다.
    return Polygon(snap_coords(points)).buffer(0)

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
                hole = random_simple_polygon(random.randint(3, 7), cx, cy, 50, 0.4)
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

        # --- 수정된 로직: 좌표 스냅핑 적용 ---
        # 입력받은 모든 좌표를 폴리곤으로 만들기 전에 스냅핑합니다.
        snapped_exteriors = [snap_coords(p) for p in exterior_paths]
        snapped_interiors = [snap_coords(p) for p in interior_paths]

        exterior_polygons = [Polygon(p) for p in snapped_exteriors if len(p) >= 3]
        total_exterior = unary_union(exterior_polygons)
        
        interior_polygons = [Polygon(p) for p in snapped_interiors if len(p) >= 3]
        total_interior = unary_union(interior_polygons)
        
        final_space = total_exterior.difference(total_interior)
        final_space = final_space.buffer(0)

        if final_space.is_empty:
             return jsonify({"error": "The final shape is empty."}), 400

        all_vertices = []
        all_segments = []
        hole_points = []
        
        def process_ring(ring):
            path = list(ring.coords)[:-1]
            path_len = len(path)
            start_idx = len(all_vertices)
            segments = [[start_idx + i, start_idx + (i + 1) % path_len] for i in range(path_len)]
            return path, segments

        polygons_to_process = list(final_space.geoms) if final_space.geom_type == 'MultiPolygon' else [final_space]

        for poly in polygons_to_process:
            if not poly.is_valid or poly.is_empty:
                continue

            new_vertices, new_segments = process_ring(poly.exterior)
            all_vertices.extend(new_vertices)
            all_segments.extend(new_segments)

            for interior_ring in poly.interiors:
                hole_poly = Polygon(interior_ring)
                if not hole_poly.is_valid or hole_poly.is_empty:
                    continue
                hole_points.append(hole_poly.representative_point().coords[0])
                new_vertices, new_segments = process_ring(interior_ring)
                all_vertices.extend(new_vertices)
                all_segments.extend(new_segments)

        if not all_vertices:
            return jsonify({"error": "No valid vertices found in the shape."}), 400

        polygon_data = dict(vertices=np.array(all_vertices), segments=np.array(all_segments))
        if hole_points:
            polygon_data['holes'] = np.array(hole_points)

        triangulation = tr.triangulate(polygon_data, 'p')
        
        if 'triangles' not in triangulation or len(triangulation['triangles']) == 0:
            return jsonify({"error": "Triangulation failed. The polygon might be too complex or degenerate."}), 500

        final_pieces = [Polygon(triangulation['vertices'][tri]) for tri in triangulation['triangles']]
        num_pieces = len(final_pieces)
        
        guard_candidates = all_vertices
        n_candidates = len(guard_candidates)

        V = np.zeros((n_candidates, num_pieces), dtype=int)
        for i, guard_pos in enumerate(guard_candidates):
            for k, piece in enumerate(final_pieces):
                if all(is_visible(guard_pos, tuple(v), final_space) for v in piece.exterior.coords[:-1]):
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
                if geom.geom_type == 'Polygon' and not geom.is_empty:
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
