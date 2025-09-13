from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box
from shapely.ops import unary_union
import pulp
# triangle 사용 안함
import traceback
import random
import math

app = Flask(__name__)
CORS(app)

# --- 유틸리티 함수 ---
def is_visible(p1, p2, polygon_space):
    """점 p1과 p2 사이의 시야가 polygon_space에 의해 막히지 않는지 확인"""
    # p1, p2는 (x,y)
    line = LineString([p1, p2])
    # 선분이 polygon_space 내부에 완전히 포함되는지 확인
    # (경계상에 있을 경우도 허용)
    try:
        return polygon_space.covers(line)
    except Exception:
        return False

def random_simple_polygon(n_vertices: int, cx=0, cy=0, avg_radius=150, irregularity=0.5):
    """더 자연스러운 형태의 무작위 단순 다각형 생성 함수"""
    if irregularity < 0: irregularity = 0
    if irregularity > 1: irregularity = 1

    angles = np.sort(np.random.uniform(0, 2 * np.pi, n_vertices))
    radii = np.random.normal(avg_radius, avg_radius * irregularity, n_vertices)
    radii = np.clip(radii, avg_radius * 0.2, avg_radius * 2)
    
    points = [(cx + r * np.cos(a), cy + r * np.sin(a)) for r, a in zip(radii, angles)]
    return Polygon(points).buffer(0)

# --- 레이캐스팅 기반 visibility polygon 계산 ---
def compute_visibility_polygon(point, space, max_dist=None, angle_eps=1e-6):
    """
    주어진 point(튜플)에서 space(Polygon or MultiPolygon) 내부로 레이들을 쏘아 visibility polygon을 근사.
    알고리즘:
      - space의 모든 꼭짓점들을 타겟으로 각도(angle)를 계산하고,
        각 angle에 대해 (angle - eps, angle, angle + eps) 세 레이를 쏴 교차점을 찾음 (collinearity 해결)
      - 각 레이에 대해 space와의 교차를 구하고, point에서 가장 가까운 교차점을 취함.
      - 교차점들을 각도순으로 정렬해 폴리곤을 만듦.
    """
    if max_dist is None:
        minx, miny, maxx, maxy = space.bounds
        diag = math.hypot(maxx - minx, maxy - miny)
        max_dist = diag * 2.0 + 1.0

    px, py = point

    # 타겟 각도 목록(모든 외부/내부 꼭짓점)
    target_points = []
    # space may be Polygon or MultiPolygon
    polys = list(space.geoms) if space.geom_type == 'MultiPolygon' else [space]
    for poly in polys:
        # exterior
        ext_coords = list(poly.exterior.coords)[:-1]
        target_points.extend(ext_coords)
        # interiors (holes)
        for ir in poly.interiors:
            target_points.extend(list(ir.coords)[:-1])

    angles = []
    for tx, ty in target_points:
        ang = math.atan2(ty - py, tx - px)
        angles.append(ang)

    # unique and sorted
    angles = sorted(set(angles))

    ray_angles = []
    for a in angles:
        ray_angles.extend([a - angle_eps, a, a + angle_eps])

    # normalize angles
    ray_angles = sorted(set(ray_angles))

    intersection_pts = []
    for a in ray_angles:
        dx = math.cos(a) * max_dist
        dy = math.sin(a) * max_dist
        far_pt = (px + dx, py + dy)
        ray = LineString([(px, py), far_pt])

        # ray와 공간의 교차 계산
        try:
            inter = ray.intersection(space)
        except Exception:
            inter = None

        chosen_point = None
        if inter is None or inter.is_empty:
            # 교차가 없으면 far_pt (하지만 보통은 공간 내부에서 끝남)
            chosen_point = far_pt
        else:
            # intersection 결과는 Point, MultiPoint, LineString, GeometryCollection 등일 수 있음
            # 가장 가까운(interior에 있는) 점을 선택: intersection을 구성하는 모든 포인트들 중에서 px,py로부터 최소 거리인 점
            candidates = []
            if inter.geom_type == 'Point':
                candidates.append((inter.x, inter.y))
            elif inter.geom_type in ['MultiPoint', 'GeometryCollection']:
                for g in inter.geoms:
                    if g.geom_type == 'Point':
                        candidates.append((g.x, g.y))
                    elif g.geom_type == 'LineString':
                        coords = list(g.coords)
                        candidates.append(coords[0])
                        candidates.append(coords[-1])
                # fallthrough
            elif inter.geom_type == 'LineString':
                coords = list(inter.coords)
                candidates.append(coords[0])
                candidates.append(coords[-1])
            else:
                # 다른 케이스들(Polygon 등)은 경계 포인트를 추출
                try:
                    for g in inter.geoms:
                        if hasattr(g, 'coords'):
                            coords = list(g.coords)
                            if coords:
                                candidates.append(coords[0])
                                candidates.append(coords[-1])
                except Exception:
                    pass

            # candidates 중 실제로 ray 상에 있고 가장 가까운 지점 선택
            min_d = None
            for cx, cy in candidates:
                # ensure point lies in direction of ray (dot product >= 0)
                vx, vy = cx - px, cy - py
                dot = vx*dx + vy*dy
                if dot < -1e-8:
                    continue
                d = math.hypot(vx, vy)
                if min_d is None or d < min_d:
                    min_d = d
                    chosen_point = (cx, cy)

            if chosen_point is None:
                # fallback: far_pt
                chosen_point = far_pt

        intersection_pts.append( (a, chosen_point) )

    # 정렬 후 폴리곤 생성 (중복 제거)
    intersection_pts = sorted([ (a,p) for a,p in intersection_pts if p is not None ], key=lambda x: x[0])
    coords = []
    seen = set()
    for a, p in intersection_pts:
        key = (round(p[0], 8), round(p[1], 8))
        if key in seen:
            continue
        seen.add(key)
        coords.append(p)

    if len(coords) < 3:
        # 시야 폴리곤을 만들 수 없으면 빈 폴리곤 반환
        return None

    try:
        poly = Polygon(coords)
        poly = poly.intersection(space)  # 혹시 공간 밖으로 나갔으면 잘라냄
        if poly.is_empty:
            return None
        return poly.buffer(0)
    except Exception:
        return None

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
        total_interior = unary_union(interior_polygons) if interior_polygons else None
        
        final_space = total_exterior.difference(total_interior) if total_interior else total_exterior
        final_space = final_space.buffer(0)

        if final_space.is_empty:
             return jsonify({"error": "The final shape is empty."}), 400

        # --- 모든 꼭짓점(후보) 추출 ---
        all_vertices = []
        polygons_to_process = list(final_space.geoms) if final_space.geom_type == 'MultiPolygon' else [final_space]
        for poly in polygons_to_process:
            ext = list(poly.exterior.coords)[:-1]
            all_vertices.extend(ext)
            for ir in poly.interiors:
                all_vertices.extend(list(ir.coords)[:-1])

        guard_candidates = all_vertices
        n_candidates = len(guard_candidates)
        if n_candidates == 0:
            return jsonify({"error": "No vertices found to place guards."}), 400

        # --- 영역을 샘플 포인트로 분해 (grid sampling) ---
        minx, miny, maxx, maxy = final_space.bounds
        # grid_resolution: 더 작을수록 더 촘촘(정밀도↑, 계산↑)
        # 기본값은 bbox의 긴 변을 80칸으로 나눈 간격
        bbox_w = maxx - minx
        bbox_h = maxy - miny
        max_dim = max(bbox_w, bbox_h)
        grid_cells = 60  # 성능/정밀도 균형: 필요시 사용자가 조정 가능
        spacing = max_dim / grid_cells if grid_cells > 0 else max_dim / 60.0

        xs = np.arange(minx + spacing/2, maxx, spacing)
        ys = np.arange(miny + spacing/2, maxy, spacing)

        sample_points = []
        for x in xs:
            for y in ys:
                pt = Point(x, y)
                if final_space.contains(pt) or final_space.touches(pt):
                    sample_points.append((x, y))

        # fallback: 샘플 포인트가 없으면 꼭짓점들을 사용
        if not sample_points:
            sample_points = guard_candidates.copy()

        m_pieces = len(sample_points)

        # --- 각 후보에 대해 visibility polygon 계산 ---
        candidate_visibility = []
        for i, g in enumerate(guard_candidates):
            vis = compute_visibility_polygon(g, final_space)
            candidate_visibility.append(vis)

        # --- 커버 관계 행렬 V (n_candidates x m_pieces) ---
        V = np.zeros((n_candidates, m_pieces), dtype=int)
        for i, vis in enumerate(candidate_visibility):
            if vis is None:
                continue
            for k, sp in enumerate(sample_points):
                # 샘플 포인트가 시야 폴리곤 내에 있으면 커버
                try:
                    if vis.contains(Point(sp)) or vis.touches(Point(sp)):
                        V[i, k] = 1
                except Exception:
                    # robust 체크
                    if vis.buffer(1e-9).contains(Point(sp)):
                        V[i, k] = 1

        # --- 선형계획(정수) 모델: 최소 개수의 guard 선택 ---
        x = pulp.LpVariable.dicts("x", range(n_candidates), cat="Binary")
        prob = pulp.LpProblem("SetCover", pulp.LpMinimize)
        prob += pulp.lpSum([x[i] for i in range(n_candidates)])

        for k in range(m_pieces):
            candidate_guards_for_piece = [x[i] for i in range(n_candidates) if V[i, k] == 1]
            if candidate_guards_for_piece:
                prob += pulp.lpSum(candidate_guards_for_piece) >= 1
            else:
                # 어떤 포인트도 커버하지 못하면 (이상치) 무시하거나 오류로 처리
                # 여기서는 안전하게 '불가능한 포인트'를 허용하지 않으려면 에러로 반환
                return jsonify({"error": f"Sample point {sample_points[k]} cannot be covered by any guard candidate."}), 500

        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        guard_indices = [i for i in range(n_candidates) if pulp.value(x[i]) and pulp.value(x[i]) > 0.5]

        guard_details = []
        for g_idx in guard_indices:
            guard_pos = guard_candidates[g_idx]
            vis = candidate_visibility[g_idx]
            vision_coords = []
            if vis is not None:
                if vis.geom_type == 'MultiPolygon':
                    for geom in vis.geoms:
                        if geom.geom_type == 'Polygon':
                            vision_coords.append(list(geom.exterior.coords))
                elif vis.geom_type == 'Polygon':
                    vision_coords.append(list(vis.exterior.coords))

            guard_details.append({"position": guard_pos, "vision_area": vision_coords})

        final_space_coords = []
        for poly in polygons_to_process:
            final_space_coords.append({
                "exterior": list(poly.exterior.coords),
                "interiors": [list(i.coords) for i in poly.interiors]
            })

        return jsonify({
            "guards": guard_details,
            "final_space": final_space_coords,
            "n_candidates": n_candidates,
            "n_sample_points": m_pieces
        })

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Calculation failed due to an internal error."}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
