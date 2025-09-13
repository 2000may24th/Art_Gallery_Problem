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
    """
    점 p1과 p2 사이의 시야가 polygon_space 내부에 완전히 포함되는지 확인합니다.
    (구멍을 통과하지 않는지 확인)
    """
    line = LineString([p1, p2])
    # contains는 라인이 도형의 경계와 교차하거나 구멍을 통과하면 False를 반환하므로 더 정확합니다.
    return polygon_space.contains(line)

def random_simple_polygon(n_vertices: int, cx=0, cy=0, avg_radius=150, irregularity=0.5, spikeyness=0.2):
    """
    더 자연스러운 형태의 무작위 단순 다각형을 생성합니다.
    """
    irregularity = np.clip(irregularity, 0, 1)
    spikeyness = np.clip(spikeyness, 0, 1)

    angles = np.sort(np.random.uniform(0, 2 * np.pi, n_vertices))
    
    # 반지름에 변동성을 추가합니다.
    radii = np.random.normal(avg_radius, avg_radius * irregularity, n_vertices)
    low_bound = avg_radius * (1 - spikeyness)
    high_bound = avg_radius * (1 + spikeyness)
    radii = np.clip(radii, low_bound, high_bound)

    points = [(cx + r * np.cos(a), cy + r * np.sin(a)) for r, a in zip(radii, angles)]
    
    # buffer(0)을 통해 유효한 폴리곤을 만듭니다.
    return Polygon(points).buffer(0)


# --- API 엔드포인트 ---

@app.route('/generate_random', methods=['GET'])
def generate_random():
    """랜덤 복합 도형 생성"""
    try:
        # 캔버스 크기에 맞게 중심점 및 반경 조정
        exterior = random_simple_polygon(random.randint(8, 15), 400, 300, 250, 0.6, 0.5)
        interiors = []
        
        min_x, min_y, max_x, max_y = exterior.bounds
        
        for _ in range(random.randint(1, 3)):
            attempts = 0
            while attempts < 50:
                # 구멍이 중앙에 더 잘 위치하도록 중심점 범위 조정
                cx = random.uniform(min_x + (max_x - min_x) * 0.2, max_x - (max_x - min_x) * 0.2)
                cy = random.uniform(min_y + (max_y - min_y) * 0.2, max_y - (max_y - min_y) * 0.2)
                
                hole = random_simple_polygon(random.randint(4, 7), cx, cy, 50, 0.4, 0.4)
                
                # 생성된 구멍이 다른 구멍과 겹치지 않고 외부에 완전히 포함되는지 확인
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
    """경비원 위치 계산"""
    try:
        data = request.get_json()
        exterior_paths = data.get('exteriors', [])
        interior_paths = data.get('interiors', [])

        if not exterior_paths:
            return jsonify({"error": "At least one exterior space is required."}), 400

        # --- Polygon 생성 ---
        exterior_polygons = [Polygon(p).buffer(0) for p in exterior_paths if len(p) >= 3]
        if not exterior_polygons:
            return jsonify({"error": "Invalid exterior path data."}), 400
        total_exterior = unary_union(exterior_polygons)

        interior_polygons = [Polygon(p).buffer(0) for p in interior_paths if len(p) >= 3]
        total_interior = unary_union(interior_polygons) if interior_polygons else None

        final_space = total_exterior.difference(total_interior) if total_interior else total_exterior
        final_space = final_space.buffer(0) # 유효성 확보

        if final_space.is_empty:
            return jsonify({"error": "The final shape is empty after subtracting interiors."}), 400

        # --- Triangle 라이브러리 입력 준비 ---
        all_vertices = []
        all_segments = []
        hole_points = []

        def add_ring_to_tri_input(coords, is_hole=False):
            """정점과 세그먼트 정보를 리스트에 추가하고, 구멍인 경우 내부 점을 추가하는 헬퍼 함수"""
            nonlocal all_vertices, all_segments, hole_points
            
            path = list(coords)
            if path and path[0] == path[-1]:
                path = path[:-1] # 마지막 중복점 제거
            
            if not path: return

            start_idx = len(all_vertices)
            all_vertices.extend(path)
            
            num_points_in_path = len(path)
            for i in range(num_points_in_path):
                all_segments.append([start_idx + i, start_idx + (i + 1) % num_points_in_path])

            if is_hole:
                # triangle 라이브러리에 이 영역이 구멍임을 알리기 위해 내부 점이 필요합니다.
                hole_poly = Polygon(path)
                # representative_point는 폴리곤 내부에 위치가 보장되는 점입니다.
                point_in_hole = hole_poly.representative_point()
                hole_points.append((point_in_hole.x, point_in_hole.y))

        # MultiPolygon일 경우를 대비하여 순회 처리
        polygons_to_process = list(final_space.geoms) if hasattr(final_space, 'geoms') else [final_space]

        for poly in polygons_to_process:
            # 1. 외부 경계선 추가
            add_ring_to_tri_input(poly.exterior.coords, is_hole=False)
            # 2. 내부 구멍(기둥) 경계선 추가
            for interior_ring in poly.interiors:
                add_ring_to_tri_input(interior_ring.coords, is_hole=True)
        
        if not all_vertices:
            return jsonify({"error": "No vertices found to process."}), 400

        polygon_data = {
            "vertices": np.array(all_vertices, dtype=float),
            "segments": np.array(all_segments, dtype=int)
        }
        if hole_points:
            polygon_data["holes"] = np.array(hole_points, dtype=float)

        # --- 삼각 분할 (Triangulation) ---
        # 'p' 옵션은 다각형을 삼각분할하고, 'q'는 품질 향상(각도 제한)을 의미합니다.
        triangulation = tr.triangulate(polygon_data, 'pq') 
        
        if 'triangles' not in triangulation or 'vertices' not in triangulation:
            return jsonify({"error": "Triangulation failed. The polygon might be too complex or invalid."}), 500

        tri_vertices = np.array(triangulation['vertices'], dtype=float)
        tri_idx = np.array(triangulation['triangles'], dtype=int)
        
        final_pieces = [Polygon(tri_vertices[tri]) for tri in tri_idx]
        final_pieces = [p for p in final_pieces if p.is_valid and final_space.contains(p.centroid)]

        if not final_pieces:
            return jsonify({"error": "No valid triangle pieces were produced after triangulation."}), 400

        # --- Set Cover 문제로 변환하여 경비원 위치 찾기 ---
        # 경비원 후보: 외부 및 내부 다각형의 모든 정점
        guard_candidates = all_vertices
        n_candidates = len(guard_candidates)
        n_pieces = len(final_pieces)

        # V[i, k] = 1: i번째 경비원 후보가 k번째 삼각형 조각을 볼 수 있음
        V = np.zeros((n_candidates, n_pieces), dtype=int)

        for i, guard_pos in enumerate(guard_candidates):
            for k, piece in enumerate(final_pieces):
                # 삼각형의 세 꼭짓점이 모두 보이면 해당 삼각형은 감시 가능하다고 판단
                if all(is_visible(guard_pos, v, final_space) for v in piece.exterior.coords[:-1]):
                    V[i, k] = 1

        # PuLP를 사용한 선형 계획법 문제 풀이
        x = pulp.LpVariable.dicts("x", range(n_candidates), cat="Binary")
        prob = pulp.LpProblem("SetCover", pulp.LpMinimize)
        prob += pulp.lpSum([x[i] for i in range(n_candidates)]) # 목적 함수: 경비원 수 최소화

        for k in range(n_pieces):
            # 제약 조건: 각 삼각형 조각은 최소 한 명의 경비원에게 감시되어야 함
            candidate_guards_for_piece = [x[i] for i in range(n_candidates) if V[i, k] == 1]
            if candidate_guards_for_piece:
                prob += pulp.lpSum(candidate_guards_for_piece) >= 1
            else:
                # 이 조각을 볼 수 있는 경비원이 한 명도 없는 경우
                return jsonify({
                    "error": f"No guard candidate can see triangle piece {k}. The area might be unreachable.",
                    "problem_piece": [list(final_pieces[k].exterior.coords)]
                }), 400
        
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status != pulp.LpStatusOptimal:
            return jsonify({"error": "Could not find an optimal solution for guard placement."}), 500

        guard_indices = [i for i in range(n_candidates) if pulp.value(x[i]) > 0.5]

        guard_details = []
        for g_idx in guard_indices:
            guard_pos = guard_candidates[g_idx]
            visible_pieces_indices = [k for k, piece in enumerate(final_pieces) if V[g_idx, k] == 1]
            visible_polygons = [final_pieces[k] for k in visible_pieces_indices]
            
            vision_area = unary_union(visible_polygons) if visible_polygons else None
            vision_coords = []
            if vision_area:
                geoms = list(vision_area.geoms) if hasattr(vision_area, 'geoms') else [vision_area]
                for geom in geoms:
                    if geom.geom_type == 'Polygon':
                        vision_coords.append(list(geom.exterior.coords))
            
            guard_details.append({"position": guard_pos, "vision_area": vision_coords})

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
