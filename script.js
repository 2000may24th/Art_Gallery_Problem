const canvas = document.getElementById('mainCanvas');
const ctx = canvas.getContext('2d');
const calculateBtn = document.getElementById('calculateBtn');
const resetBtn = document.getElementById('resetBtn');
const randomBtn = document.getElementById('randomBtn');
const finishPathBtn = document.getElementById('finishPathBtn');
const statusElem = document.getElementById('status');
const modeRadios = document.querySelectorAll('input[name="mode"]');

// 상태 관리 변수
let currentPath = [];
let exteriorPaths = [];
let interiorPaths = [];
let resultData = null;

function getCurrentMode() {
    return document.querySelector('input[name="mode"]:checked').value;
}

// 전체 다시 그리기 함수
function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // 1. 확정된 외부 공간 그리기
    ctx.fillStyle = 'rgba(220, 220, 220, 0.7)'; // 밝은 회색
    exteriorPaths.forEach(path => {
        ctx.beginPath();
        ctx.moveTo(path[0][0], path[0][1]);
        for (let i = 1; i < path.length; i++) ctx.lineTo(path[i][0], path[i][1]);
        ctx.closePath();
        ctx.fill();
    });

    // 2. 확정된 내부 공간(기둥) 그리기
    ctx.fillStyle = 'white'; // 캔버스 배경색과 동일
    interiorPaths.forEach(path => {
        ctx.beginPath();
        ctx.moveTo(path[0][0], path[0][1]);
        for (let i = 1; i < path.length; i++) ctx.lineTo(path[i][0], path[i][1]);
        ctx.closePath();
        ctx.fill();
        ctx.strokeStyle = '#aaa';
        ctx.stroke();
    });

    // 3. 현재 그리고 있는 경로 그리기
    if (currentPath.length > 0) {
        ctx.beginPath();
        ctx.moveTo(currentPath[0][0], currentPath[0][1]);
        for (let i = 1; i < currentPath.length; i++) ctx.lineTo(currentPath[i][0], currentPath[i][1]);
        if (currentPath.length > 2) { // 닫히는 선은 점선으로 표시
            ctx.save();
            ctx.setLineDash([5, 5]);
            ctx.lineTo(currentPath[0][0], currentPath[0][1]);
            ctx.strokeStyle = 'grey';
            ctx.stroke();
            ctx.restore();
        } else {
            ctx.strokeStyle = 'black';
            ctx.stroke();
        }
        // 꼭짓점 그리기
        currentPath.forEach(v => {
            ctx.beginPath(); ctx.arc(v[0], v[1], 4, 0, 2 * Math.PI);
            ctx.fillStyle = 'blue'; ctx.fill();
        });
    }

    // 4. 계산 결과가 있으면 그리기
    if (resultData) drawResults(resultData);
}

// 캔버스 클릭 이벤트
canvas.addEventListener('click', (event) => {
    const rect = canvas.getBoundingClientRect();
    currentPath.push([event.clientX - rect.left, event.clientY - rect.top]);
    resultData = null; // 그림을 수정하면 이전 결과 초기화
    draw();
});

// 현재 도형 확정 버튼
finishPathBtn.addEventListener('click', () => {
    if (currentPath.length < 3) {
        alert("최소 3개 이상의 꼭짓점이 필요합니다.");
        return;
    }
    if (getCurrentMode() === 'exterior') {
        exteriorPaths.push(currentPath);
    } else {
        interiorPaths.push(currentPath);
    }
    currentPath = [];
    draw();
});

// 전체 초기화 버튼
resetBtn.addEventListener('click', () => {
    currentPath = [];
    exteriorPaths = [];
    interiorPaths = [];
    resultData = null;
    statusElem.textContent = "모드를 선택하고 캔버스에 클릭하여 도형을 그리세요.";
    draw();
});

// 랜덤 생성 버튼
randomBtn.addEventListener('click', async () => {
    resetBtn.click(); // 모든 것 초기화
    statusElem.textContent = "랜덤 도형 생성 중...";
    try {
        const response = await fetch('https://art-gallery-problem.onrender.com/generate_random');
        if (!response.ok) throw new Error('서버에서 랜덤 도형을 생성하지 못했습니다.');
        const data = await response.json();
        exteriorPaths = data.exteriors;
        interiorPaths = data.interiors;
        statusElem.textContent = "랜덤 도형이 생성되었습니다. 계산 버튼을 누르세요.";
        draw();
    } catch (error) {
        statusElem.textContent = `오류: ${error.message}`;
    }
});

// 계산 버튼
calculateBtn.addEventListener('click', async () => {
    if (exteriorPaths.length === 0 && currentPath.length < 3) {
        alert("최소 1개 이상의 외부 공간이 필요합니다.");
        return;
    }
    // 현재 그리고 있는 경로가 있으면 자동으로 확정
    if (currentPath.length >= 3) finishPathBtn.click();

    statusElem.textContent = "계산 중... 복잡한 도형은 시간이 걸릴 수 있습니다.";
    try {
        const response = await fetch('https://art-gallery-problem.onrender.com/calculate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ exteriors: exteriorPaths, interiors: interiorPaths }),
        });
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error || '알 수 없는 서버 오류');
        }
        resultData = await response.json();
        statusElem.textContent = `계산 완료! ${resultData.guards.length}명의 경비원이 배치되었습니다.`;
        draw();
    } catch (error) {
        statusElem.textContent = `계산 실패: ${error.message}`;
        resultData = null;
    }
});

function drawResults(result) {
    // 최종 공간의 외곽선 다시 그리기 (계산 후 모습)
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 2.5;
    result.final_space.forEach(space => {
        ctx.beginPath();
        ctx.moveTo(space.exterior[0][0], space.exterior[0][1]);
        for(let i=1; i<space.exterior.length; i++) ctx.lineTo(space.exterior[i][0], space.exterior[i][1]);
        ctx.closePath();
        space.interiors.forEach(hole => {
            ctx.moveTo(hole[0][0], hole[0][1]);
            for(let i=1; i<hole.length; i++) ctx.lineTo(hole[i][0], hole[i][1]);
            ctx.closePath();
        });
        ctx.stroke();
    });

    const colors = ['rgba(255, 99, 132, 0.4)', 'rgba(54, 162, 235, 0.4)', 'rgba(255, 206, 86, 0.4)', 'rgba(75, 192, 192, 0.4)', 'rgba(153, 102, 255, 0.4)'];

    // 각 경비원의 감시 영역 그리기
    result.guards.forEach((guard, index) => {
        ctx.fillStyle = colors[index % colors.length];
        guard.vision_area.forEach(polyCoords => {
            ctx.beginPath();
            ctx.moveTo(polyCoords[0][0], polyCoords[0][1]);
            for (let i = 1; i < polyCoords.length; i++) ctx.lineTo(polyCoords[i][0], polyCoords[i][1]);
            ctx.closePath();
            ctx.fill();
        });
    });

    // 경비원 위치 그리기
    result.guards.forEach((guard, index) => {
        drawStar(guard.position[0], guard.position[1], 5, 12, 6, colors[index % colors.length].replace('0.4', '1'));
    });
}

function drawStar(cx, cy, spikes, outerRadius, innerRadius, color) { /* 이전과 동일 */ 
    let rot = Math.PI / 2 * 3;
    let x = cx; let y = cy;
    let step = Math.PI / spikes;
    ctx.beginPath(); ctx.moveTo(cx, cy - outerRadius);
    for (let i = 0; i < spikes; i++) {
        x = cx + Math.cos(rot) * outerRadius; y = cy + Math.sin(rot) * outerRadius; ctx.lineTo(x, y); rot += step;
        x = cx + Math.cos(rot) * innerRadius; y = cy + Math.sin(rot) * innerRadius; ctx.lineTo(x, y); rot += step;
    }
    ctx.lineTo(cx, cy - outerRadius); ctx.closePath();
    ctx.lineWidth = 2; ctx.strokeStyle = 'black'; ctx.stroke();
    ctx.fillStyle = color; ctx.fill();

}
