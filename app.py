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

# --- 유틸리티 함수 ---
def is_visible(p1, p2, polygon_space: Polygon | MultiPolygon):
    """
    점 p1과 p2 사이의 시야가 polygon_space 내부에 완전히 포함되는지 확인합니다.
    기존의 covers는 경계에 닿는 것을 허용하지 않아 문제가 될 수 있으므로 contains로 변경합니다.
    """
    line = LineString([p1, p2])
    # contains는 선이 공간 내부에 있거나 경계에 닿는 것을 허용하여 더 안정적입니다.
    return polygon_space.contains(line)

def random_simple_polygon(n_vertices: int, cx=0, cy=0, avg_radius=150, irregularity=0.5):
    """자연스러운 무작위 단순 다각형 생성"""
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
    """랜덤 복합 도형 생성"""
    try:
        exterior = random_simple_polygon(random.randint(8, 15), 400, 300, 250, 0.6)
        interiors = []
        for _ in range(random.randint(1, 3)):
            min_x, min_y, max_x, max_y = exterior.bounds
            attempts = 0
            while attempts < 50:
                cx, cy = random.uniform(min_x, max_x), random.uniform(min_y, max_y)
                hole = random_simple_polygon(random.randint(4, 7), cx, cy, 50, 0.4)
                # 생성된 구멍이 다른 구멍과 겹치지 않고 외부에 포함되는지 확인
                if exterior.contains(hole) and not any(h.intersects(hole) for h in interiors):
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

        # --- Polygon 생성 ---
        exterior_polygons = [Polygon(p).buffer(0) for p in exterior_paths if len(p) >= 3]
        total_exterior = unary_union(exterior_polygons)

        interior_polygons = [Polygon(p).buffer(0) for p in interior_paths if len(p) >= 3]
        total_interior = unary_union(interior_polygons) if interior_polygons else None

        final_space = total_exterior.difference(total_interior) if total_interior else total_exterior
        final_space = final_space.buffer(0)

        if final_space.is_empty:
            return jsonify({"error": "The final shape is empty."}), 400

        # --- Triangle 입력 준비 ---
        all_vertices = []
        all_segments = []
        holes_for_triangulation = []

        def process_exterior_ring(coords):
            nonlocal all_vertices, all_segments
            path = list(coords)[:-1] if coords[0] == coords[-1] else list(coords)
            start_idx = len(all_vertices)
            all_vertices.extend(path)
            for i in range(len(path)):
                all_segments.append([start_idx + i, start_idx + (i + 1) % len(path)])

        polygons_to_process = list(final_space.geoms) if final_space.geom_type == "MultiPolygon" else [final_space]

        for poly in polygons_to_process:
            process_exterior_ring(poly.exterior.coords)
            for interior_ring in poly.interiors:
                # ★★★★★ 주요 수정 사항 1 ★★★★★
                # 구멍(hole)을 삼각 분할 라이브러리에 전달하려면, 구멍의 내부에 있는 점을 지정해야 합니다.
                # 기존 코드는 'final_space'에 점이 포함되는지 잘못 확인하고 있었습니다.
                # 구멍은 final_space에서 제외된 영역이므로 이 조건은 항상 거짓이 됩니다.
                # 따라서 조건 확인 없이 각 구멍 내부의 대표점만 수집하면 됩니다.
                hole_poly = Polygon(interior_ring)
                holes_for_triangulation.append(hole_poly.representative_point().coords[0])

        polygon_data = {
            "vertices": np.array(all_vertices, dtype=float),
            "segments": np.array(all_segments, dtype=int)
        }
        if holes_for_triangulation:
            polygon_data["holes"] = np.array(holes_for_triangulation, dtype=float)

        # --- Triangulate ---
        # 'p' 옵션은 다각형을 삼각분할하며, 'q'는 품질을 개선합니다.
        triangulation = tr.triangulate(polygon_data, 'pq')
        if 'triangles' not in triangulation or 'vertices' not in triangulation:
            return jsonify({"error": "Triangulation failed. The input shape might be too complex or invalid."}), 400

        tri_vertices = np.array(triangulation['vertices'], dtype=float)
        tri_idx = np.array(triangulation['triangles'], dtype=int)
        # 생성된 삼각형이 유효하고, 최종 공간 내부에 있는지 다시 한번 확인합니다.
        final_pieces = [Polygon(tri_vertices[tri]) for tri in tri_idx if final_space.contains(Polygon(tri_vertices[tri]).centroid)]

        if not final_pieces:
            return jsonify({"error": "No valid triangle pieces were produced after triangulation."}), 400

        # --- Set Cover (guards) ---
        guard_candidates = all_vertices
        n_candidates = len(guard_candidates)
        V = np.zeros((n_candidates, len(final_pieces)), dtype=int)

        for i, guard_pos in enumerate(guard_candidates):
            for k, piece in enumerate(final_pieces):
                # 감시자가 삼각형의 모든 꼭짓점을 볼 수 있다면, 그 삼각형 전체를 볼 수 있다고 가정합니다.
                if all(is_visible(guard_pos, v, final_space) for v in piece.exterior.coords[:-1]):
                    V[i, k] = 1

        x = pulp.LpVariable.dicts("x", range(n_candidates), cat="Binary")
        prob = pulp.LpProblem("SetCover", pulp.LpMinimize)
        prob += pulp.lpSum([x[i] for i in range(n_candidates)])

        for k in range(len(final_pieces)):
            candidate_guards_for_piece = [x[i] for i in range(n_candidates) if V[i, k] == 1]
            if candidate_guards_for_piece:
                prob += pulp.lpSum(candidate_guards_for_piece) >= 1
            else:
                # 이 삼각형을 볼 수 있는 감시자가 없는 경우, 에러 대신 경고를 로깅하고 계속 진행할 수 있습니다.
                # 하지만 현재로서는 계산 실패로 처리하는 것이 명확합니다.
                print(f"Warning: No guard candidate can see triangle {k}")
                return jsonify({"error": f"Calculation failed: A piece of the area (triangle {k}) is not visible from any vertex."}), 400

        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        guard_indices = [i for i in range(n_candidates) if pulp.value(x[i]) > 0.5]

        guard_details = []
        for g_idx in guard_indices:
            guard_pos = guard_candidates[g_idx]
            visible_pieces = [p for k, p in enumerate(final_pieces) if V[g_idx, k] == 1]
            vision_area = unary_union(visible_pieces) if visible_pieces else None
            vision_coords = []
            if vision_area and not vision_area.is_empty:
                geoms = list(vision_area.geoms) if vision_area.geom_type == "MultiPolygon" else [vision_area]
                for geom in geoms:
                    if geom.geom_type == 'Polygon':
                        vision_coords.append(list(geom.exterior.coords))
            guard_details.append({"position": list(guard_pos), "vision_area": vision_coords})

        # --- final_space 배열 형태로 전송 ---
        final_space_list = []
        for poly in polygons_to_process:
            final_space_list.append({
                "exterior": list(poly.exterior.coords),
                "interiors": [list(i.coords) for i in poly.interiors]
            })

        return jsonify({
            "guards": guard_details,
            "final_space": final_space_list
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
