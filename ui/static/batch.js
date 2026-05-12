/**
 * 批量识别模块
 * 多图上传 + 结果表格 + ECharts 统计面板 + 导出 CSV
 */

let batchResults = []; // 保存当前批量结果，用于导出

document.addEventListener("DOMContentLoaded", () => {
    initBatchUpload();
    initBatchExport();
});

function initBatchUpload() {
    const area = document.getElementById("batchUploadArea");
    const input = document.getElementById("batchFileInput");

    if (!area || !input) return;

    area.addEventListener("click", () => input.click());

    area.addEventListener("dragover", (e) => {
        e.preventDefault();
        area.classList.add("drag-over");
    });

    area.addEventListener("dragleave", () => area.classList.remove("drag-over"));

    area.addEventListener("drop", (e) => {
        e.preventDefault();
        area.classList.remove("drag-over");
        if (e.dataTransfer.files.length > 0) {
            runBatchPredict(e.dataTransfer.files);
        }
    });

    input.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            runBatchPredict(e.target.files);
            input.value = "";
        }
    });
}

async function runBatchPredict(files) {
    if (files.length === 0) return;
    showLoading(`正在批量识别 ${files.length} 张图片...`);

    const formData = new FormData();
    for (const f of files) {
        formData.append("files", f);
    }

    try {
        const resp = await fetch("/api/batch-predict", {
            method: "POST",
            body: formData,
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || "批量识别失败");
        }

        const data = await resp.json();
        batchResults = data.results;
        displayBatchResult(data);
    } catch (e) {
        alert("批量识别失败: " + e.message);
    } finally {
        hideLoading();
    }
}

function displayBatchResult(data) {
    const { results, stats, model_name } = data;

    // 统计摘要
    document.getElementById("batchStatsBar").style.display = "flex";
    document.getElementById("batchStatTotal").textContent = stats.total;
    document.getElementById("batchStatAvgConf").textContent = (stats.average_confidence * 100).toFixed(1) + "%";
    document.getElementById("batchStatLowConf").textContent = stats.low_confidence_count;

    // 结果表格
    document.getElementById("batchResultCard").style.display = "block";
    const tbody = document.getElementById("batchResultBody");
    tbody.innerHTML = "";

    results.forEach((r, i) => {
        if (r.error) {
            const tr = document.createElement("tr");
            tr.innerHTML = `<td>${i + 1}</td><td>${r.filename}</td><td colspan="2" style="color:#DC2626">${r.error}</td>`;
            tbody.appendChild(tr);
            return;
        }

        const color = CLASS_COLORS[r.class_name] || "#6B7280";
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td>${i + 1}</td>
            <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${r.filename}</td>
            <td><span class="batch-class-tag" style="background:${color}">${r.class_name}</span></td>
            <td>${(r.confidence * 100).toFixed(1)}%</td>
        `;
        tbody.appendChild(tr);
    });

    // ECharts 图表
    renderBatchCharts(stats);
}

function renderBatchCharts(stats) {
    document.getElementById("batchChartsRow").style.display = "grid";

    // 饼图：各类别占比
    const pieDom = document.getElementById("batchPieChart");
    const pieChart = echarts.init(pieDom);

    const pieData = Object.entries(stats.category_counts).map(([name, count]) => ({
        name: name,
        value: count,
        itemStyle: { color: CLASS_COLORS[name] || "#94A3B8" },
    }));

    pieChart.setOption({
        title: { text: "各类别占比", left: "center", textStyle: { fontSize: 14, color: "#1E293B" } },
        tooltip: { trigger: "item", formatter: "{b}: {c} ({d}%)" },
        series: [{
            type: "pie",
            radius: ["35%", "65%"],
            center: ["50%", "55%"],
            label: { formatter: "{b}\n{d}%", fontSize: 12 },
            data: pieData,
        }],
    });

    // 柱状图：各类别数量
    const barDom = document.getElementById("batchBarChart");
    const barChart = echarts.init(barDom);

    const categories = Object.keys(stats.category_counts);
    const counts = Object.values(stats.category_counts);
    const barColors = categories.map((c) => CLASS_COLORS[c] || "#94A3B8");

    barChart.setOption({
        title: { text: "各类别数量", left: "center", textStyle: { fontSize: 14, color: "#1E293B" } },
        tooltip: { trigger: "axis" },
        xAxis: { type: "category", data: categories, axisLabel: { fontSize: 12 } },
        yAxis: { type: "value", minInterval: 1 },
        series: [{
            type: "bar",
            data: counts.map((v, i) => ({ value: v, itemStyle: { color: barColors[i] } })),
            barWidth: "40%",
            label: { show: true, position: "top" },
        }],
    });

    // 响应式
    window.addEventListener("resize", () => {
        pieChart.resize();
        barChart.resize();
    });
}

// ============================================================
// 导出 CSV
// ============================================================
function initBatchExport() {
    const btn = document.getElementById("batchExportBtn");
    if (!btn) return;

    btn.addEventListener("click", () => {
        if (batchResults.length === 0) {
            alert("没有可导出的数据");
            return;
        }

        const header = "序号,文件名,分类结果,置信度\n";
        const rows = batchResults.map((r, i) => {
            if (r.error) return `${i + 1},${r.filename},识别失败,0`;
            return `${i + 1},${r.filename},${r.class_name},${(r.confidence * 100).toFixed(1)}%`;
        }).join("\n");

        const csv = "\uFEFF" + header + rows; // BOM for Excel 中文兼容
        const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
        const url = URL.createObjectURL(blob);

        const a = document.createElement("a");
        a.href = url;
        a.download = "batch_results_" + new Date().toISOString().slice(0, 10) + ".csv";
        a.click();
        URL.revokeObjectURL(url);
    });
}
