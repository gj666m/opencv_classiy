/**
 * 模型对比模块
 * 上传一张图，同时调用 4 个模型推理，展示四宫格 + 表格
 */

document.addEventListener("DOMContentLoaded", () => {
    initCompareUpload();
});

function initCompareUpload() {
    const area = document.getElementById("compareUploadArea");
    const input = document.getElementById("compareFileInput");

    if (!area || !input) return;

    setupDragDrop(area, input);

    input.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            runCompare(e.target.files[0]);
            input.value = "";
        }
    });
}

async function runCompare(file) {
    showLoading("四模型对比推理中...");

    const formData = new FormData();
    formData.append("file", file);

    try {
        const resp = await fetch("/api/compare", {
            method: "POST",
            body: formData,
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || "对比推理失败");
        }

        const data = await resp.json();
        displayCompareResult(data.results, file);
    } catch (e) {
        alert("对比推理失败: " + e.message);
    } finally {
        hideLoading();
    }
}

function displayCompareResult(results, file) {
    // 隐藏上传区，显示结果
    document.getElementById("compareUploadArea").style.display = "none";
    document.getElementById("compareResultArea").style.display = "block";

    // 找出最高置信度的模型
    let bestIdx = 0;
    let bestConf = 0;
    results.forEach((r, i) => {
        if (r.confidence > bestConf) {
            bestConf = r.confidence;
            bestIdx = i;
        }
    });

    // 四宫格
    const grid = document.getElementById("compareGrid");
    grid.innerHTML = "";
    results.forEach((r, i) => {
        const color = CLASS_COLORS[r.class_name] || "#6B7280";
        const isBest = i === bestIdx;
        const item = document.createElement("div");
        item.className = "compare-item" + (isBest ? " best" : "");
        item.innerHTML = `
            <div class="compare-model-name">${r.model_name}</div>
            <div class="compare-class" style="color:${color}">${r.class_name}</div>
            <div class="compare-conf">${(r.confidence * 100).toFixed(1)}%</div>
            <div class="compare-time">⏱ ${r.inference_time_ms} ms</div>
        `;
        grid.appendChild(item);
    });

    // 表格
    const tbody = document.getElementById("compareTableBody");
    tbody.innerHTML = "";
    results.forEach((r, i) => {
        const color = CLASS_COLORS[r.class_name] || "#6B7280";
        const isBest = i === bestIdx;
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td class="${isBest ? "best-cell" : ""}">${r.model_name}</td>
            <td style="color:${color};font-weight:600">${r.class_name}</td>
            <td class="${isBest ? "best-cell" : ""}">${(r.confidence * 100).toFixed(1)}%</td>
            <td>${r.inference_time_ms} ms</td>
        `;
        tbody.appendChild(tr);
    });

    // 添加重新上传提示（点击结果区域头部）
    const resultArea = document.getElementById("compareResultArea");
    // 在最前面添加一个操作栏（如果还没有的话）
    if (!resultArea.querySelector(".compare-actions")) {
        const actions = document.createElement("div");
        actions.className = "compare-actions";
        actions.style.cssText = "display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;";
        actions.innerHTML = `
            <span style="font-size:14px;color:var(--text-secondary)">四模型对比结果（最高置信度已高亮）</span>
            <button class="btn btn-outline" id="compareRetryBtn">重新选择图片</button>
        `;
        resultArea.insertBefore(actions, resultArea.firstChild);

        document.getElementById("compareRetryBtn").addEventListener("click", () => {
            document.getElementById("compareResultArea").style.display = "none";
            document.getElementById("compareUploadArea").style.display = "flex";
            // 移除操作栏
            actions.remove();
        });
    }
}
