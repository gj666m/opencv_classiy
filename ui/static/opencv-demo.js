/**
 * OpenCV 演示模块
 * 上传图片 → 调 /api/opencv-pipeline → 横向流水线展示四步结果
 */

document.addEventListener("DOMContentLoaded", () => {
    initOpenCVUpload();
    initImageModal();
});

function initOpenCVUpload() {
    const area = document.getElementById("opencvUploadArea");
    const input = document.getElementById("opencvFileInput");

    if (!area || !input) return;

    setupDragDrop(area, input);

    input.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            runOpenCVPipeline(e.target.files[0]);
            input.value = "";
        }
    });
}

async function runOpenCVPipeline(file) {
    showLoading("正在生成预处理管线...");

    const formData = new FormData();
    formData.append("file", file);

    try {
        const resp = await fetch("/api/opencv-pipeline", {
            method: "POST",
            body: formData,
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || "管线生成失败");
        }

        const data = await resp.json();
        displayPipeline(data.steps);
    } catch (e) {
        alert("管线生成失败: " + e.message);
    } finally {
        hideLoading();
    }
}

function displayPipeline(steps) {
    // 显示结果区
    document.getElementById("pipelineResultArea").style.display = "block";

    const container = document.getElementById("pipelineSteps");
    container.innerHTML = "";

    steps.forEach((step, i) => {
        // 图片步骤
        const stepDiv = document.createElement("div");
        stepDiv.className = "pipeline-step";
        stepDiv.innerHTML = `
            <img class="pipeline-step-img" src="data:image/jpeg;base64,${step.image_base64}" alt="${step.name}" data-full="${step.image_base64}">
            <div class="pipeline-step-name">${step.name}</div>
            <div class="pipeline-step-desc">${step.description || ""}</div>
        `;
        container.appendChild(stepDiv);

        // 箭头（最后一个不加）
        if (i < steps.length - 1) {
            const arrow = document.createElement("div");
            arrow.className = "pipeline-arrow";
            arrow.innerHTML = '<i class="ri-arrow-right-s-line"></i>';
            container.appendChild(arrow);
        }
    });
}

// ============================================================
// 图片点击放大
// ============================================================
function initImageModal() {
    const modal = document.getElementById("imageModal");
    const modalImg = document.getElementById("imageModalImg");

    // 点击管线图片放大
    document.addEventListener("click", (e) => {
        if (e.target.classList.contains("pipeline-step-img")) {
            const base64 = e.target.dataset.full;
            if (base64) {
                modalImg.src = "data:image/jpeg;base64," + base64;
                modal.classList.add("active");
            }
        }
    });

    // 点击模态关闭
    modal.addEventListener("click", () => {
        modal.classList.remove("active");
    });
}
