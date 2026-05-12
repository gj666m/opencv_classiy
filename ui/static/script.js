/**
 * 垃圾智能分类系统 — 主交互逻辑
 * 负责：Tab 切换、首页上传、单图识别、模型选择、历史联动
 */

// 四大类颜色映射（全局，供其他模块使用）
const CLASS_COLORS = {
    "可回收物": "#2563EB",
    "厨余垃圾": "#16A34A",
    "有害垃圾": "#DC2626",
    "其他垃圾": "#6B7280",
};

// ============================================================
// DOM 引用
// ============================================================
const loadingOverlay = document.getElementById("loadingOverlay");
const loadingText = document.getElementById("loadingText");
const modelSelect = document.getElementById("modelSelect");
const modelDesc = document.getElementById("modelDesc");

// ============================================================
// 初始化
// ============================================================
document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initSubTabs();
    loadModels();
    initHomeUpload();
    initSingleUpload();
    initModelSwitch();
    initClearHistory();
    History.render("historyGrid");
});

// ============================================================
// Tab 切换
// ============================================================
function initTabs() {
    document.querySelectorAll(".tab-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            // 切换按钮状态
            document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");

            // 切换内容
            const tabId = "tab-" + btn.dataset.tab;
            document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
            document.getElementById(tabId).classList.add("active");

            // 批量识别页：图表可能需要 resize
            if (btn.dataset.tab === "batch") {
                window.dispatchEvent(new Event("resize"));
            }
        });
    });
}

// 子 Tab 切换（智能识别页内的 单图/对比）
function initSubTabs() {
    document.querySelectorAll(".sub-tab-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            const parent = btn.closest(".recognize-main");
            parent.querySelectorAll(".sub-tab-btn").forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");

            const subId = "subtab-" + btn.dataset.subtab;
            parent.querySelectorAll(".sub-tab-content").forEach((c) => c.classList.remove("active"));
            document.getElementById(subId).classList.add("active");
        });
    });
}

// ============================================================
// 模型加载与切换
// ============================================================
async function loadModels() {
    try {
        const resp = await fetch("/api/models");
        const data = await resp.json();

        modelSelect.innerHTML = "";
        data.models.forEach((m) => {
            const opt = document.createElement("option");
            opt.value = m.key;
            opt.textContent = m.name;
            opt.disabled = !m.available;
            if (m.current) opt.selected = true;
            modelSelect.appendChild(opt);
        });

        updateModelDesc(data);
    } catch {
        modelSelect.innerHTML = '<option value="">加载失败</option>';
    }
}

function updateModelDesc(data) {
    const current = data.models.find((m) => m.current);
    if (current) {
        modelDesc.textContent = current.description;
    }
}

function initModelSwitch() {
    modelSelect.addEventListener("change", async () => {
        const model = modelSelect.value;
        if (!model) return;
        try {
            const resp = await fetch("/api/switch-model", {
                method: "POST",
                headers: { "Content-Type": "application/x-www-form-urlencoded" },
                body: "model=" + encodeURIComponent(model),
            });
            const data = await resp.json();
            const modelsResp = await fetch("/api/models");
            const modelsData = await modelsResp.json();
            updateModelDesc(modelsData);
        } catch (e) {
            console.error("切换模型失败:", e);
        }
    });
}

// ============================================================
// 首页快速上传
// ============================================================
function initHomeUpload() {
    const area = document.getElementById("homeUploadArea");
    const input = document.getElementById("homeFileInput");
    setupDragDrop(area, input);

    input.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            // 切换到智能识别 Tab，用单图识别
            switchToTab("recognize");
            uploadAndPredict(e.target.files[0]);
            input.value = "";
        }
    });
}

// ============================================================
// 单图识别上传
// ============================================================
function initSingleUpload() {
    const area = document.getElementById("singleUploadArea");
    const input = document.getElementById("singleFileInput");
    setupDragDrop(area, input);

    input.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            uploadAndPredict(e.target.files[0]);
            input.value = "";
        }
    });

    // 重新上传
    document.getElementById("singleReuploadBtn").addEventListener("click", () => {
        document.getElementById("singleResultCard").style.display = "none";
        area.style.display = "flex";
    });
}

// ============================================================
// 上传并识别（单图）
// ============================================================
async function uploadAndPredict(file) {
    showLoading("正在识别中...");

    // 隐藏上传区，显示结果区
    document.getElementById("singleUploadArea").style.display = "none";
    document.getElementById("singleResultCard").style.display = "flex";

    const formData = new FormData();
    formData.append("file", file);

    const selectedModel = modelSelect.value;
    if (selectedModel) {
        formData.append("model", selectedModel);
    }

    try {
        const resp = await fetch("/api/predict", {
            method: "POST",
            body: formData,
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || "识别失败");
        }

        const result = await resp.json();
        displaySingleResult(result, file);
    } catch (e) {
        alert("识别失败: " + e.message);
        document.getElementById("singleResultCard").style.display = "none";
        document.getElementById("singleUploadArea").style.display = "flex";
    } finally {
        hideLoading();
    }
}

// ============================================================
// 显示单图识别结果
// ============================================================
function displaySingleResult(result, file) {
    // 标注图
    if (result.annotated_image) {
        document.getElementById("singleAnnotatedImg").src = "data:image/jpeg;base64," + result.annotated_image;
    }

    // 分类结果
    const color = CLASS_COLORS[result.class_name] || "#333";
    document.getElementById("singleResultClass").textContent = result.class_name;
    document.getElementById("singleResultClass").style.color = color;
    document.getElementById("singleResultConf").textContent = "置信度: " + (result.confidence * 100).toFixed(1) + "%";

    // 结果卡片背景
    document.getElementById("singleResultBox").style.background = hexToGradient(color);

    // 概率分布
    renderProbChart("singleProbChart", result.probabilities);

    // 投放建议
    if (result.advice) {
        document.getElementById("singleAdviceText").textContent = result.advice;
        document.getElementById("singleAdviceCard").style.borderLeftColor = color;
    }

    // 模型信息
    document.getElementById("singleModelInfo").textContent = "模型: " + result.model_name;

    // 保存到历史
    saveToHistory(result, file);
}

// ============================================================
// 概率分布柱状图
// ============================================================
function renderProbChart(containerId, probabilities) {
    const container = document.getElementById(containerId);
    container.innerHTML = "";

    for (const [name, prob] of Object.entries(probabilities)) {
        const color = CLASS_COLORS[name] || "#94A3B8";
        const pct = (prob * 100).toFixed(1);

        const item = document.createElement("div");
        item.className = "prob-bar-item";
        item.innerHTML = `
            <span class="prob-bar-label">${name}</span>
            <div class="prob-bar-track">
                <div class="prob-bar-fill" style="width: ${pct}%; background: ${color};"></div>
            </div>
            <span class="prob-bar-value">${pct}%</span>
        `;
        container.appendChild(item);
    }
}

// ============================================================
// 历史记录保存
// ============================================================
function saveToHistory(result, file) {
    // 用 FileReader 从原始文件生成缩略图（避免 annotated 图未加载完的黑框问题）
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            const canvas = document.createElement("canvas");
            const ctx = canvas.getContext("2d");
            const thumbSize = 64;
            canvas.width = thumbSize;
            canvas.height = thumbSize;
            // 居中裁切为正方形
            const minSide = Math.min(img.width, img.height);
            const sx = (img.width - minSide) / 2;
            const sy = (img.height - minSide) / 2;
            ctx.drawImage(img, sx, sy, minSide, minSide, 0, 0, thumbSize, thumbSize);
            const thumbnail = canvas.toDataURL("image/jpeg", 0.6);

            History.add({
                class_name: result.class_name,
                confidence: result.confidence,
                thumbnail: thumbnail,
                probabilities: result.probabilities,
                model_name: result.model_name,
                advice: result.advice || "",
            });

            History.render("historyGrid");
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

// 历史记录点击：回看
window.addEventListener("history-select", (e) => {
    const item = e.detail;
    // 在单图结果中展示（使用缩略图）
    document.getElementById("singleUploadArea").style.display = "none";
    document.getElementById("singleResultCard").style.display = "flex";

    document.getElementById("singleAnnotatedImg").src = item.thumbnail;
    const color = CLASS_COLORS[item.class_name] || "#333";
    document.getElementById("singleResultClass").textContent = item.class_name;
    document.getElementById("singleResultClass").style.color = color;
    document.getElementById("singleResultConf").textContent = "置信度: " + (item.confidence * 100).toFixed(1) + "%";
    document.getElementById("singleResultBox").style.background = hexToGradient(color);
    renderProbChart("singleProbChart", item.probabilities);

    // 更新投放建议
    if (item.advice) {
        document.getElementById("singleAdviceText").textContent = item.advice;
        document.getElementById("singleAdviceCard").style.borderLeftColor = color;
    }

    document.getElementById("singleModelInfo").textContent = "模型: " + item.model_name;
});

// 清空历史
function initClearHistory() {
    document.getElementById("clearHistoryBtn").addEventListener("click", () => {
        if (confirm("确定要清空所有识别历史吗？")) {
            History.clear();
            History.render("historyGrid");
        }
    });
}

// ============================================================
// 通用工具函数
// ============================================================
function setupDragDrop(area, input) {
    area.addEventListener("click", () => input.click());

    area.addEventListener("dragover", (e) => {
        e.preventDefault();
        area.classList.add("drag-over");
    });

    area.addEventListener("dragleave", () => {
        area.classList.remove("drag-over");
    });

    area.addEventListener("drop", (e) => {
        e.preventDefault();
        area.classList.remove("drag-over");
        if (e.dataTransfer.files.length > 0) {
            // 触发 change 事件
            const dt = new DataTransfer();
            dt.items.add(e.dataTransfer.files[0]);
            input.files = dt.files;
            input.dispatchEvent(new Event("change"));
        }
    });
}

function switchToTab(tabName) {
    document.querySelectorAll(".tab-btn").forEach((btn) => {
        if (btn.dataset.tab === tabName) {
            btn.click();
        }
    });
}

function showLoading(text) {
    loadingText.textContent = text || "正在识别中...";
    loadingOverlay.classList.add("active");
}

function hideLoading() {
    loadingOverlay.classList.remove("active");
}

function hexToGradient(hex) {
    return `linear-gradient(135deg, ${hex}15, ${hex}25)`;
}
