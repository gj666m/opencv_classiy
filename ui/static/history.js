/**
 * 识别历史管理模块 — localStorage 封装
 */

const HISTORY_KEY = "waste_history";
const HISTORY_MAX = 20;

const History = {
    /**
     * 获取所有历史记录
     */
    getAll() {
        try {
            const data = localStorage.getItem(HISTORY_KEY);
            return data ? JSON.parse(data) : [];
        } catch {
            return [];
        }
    },

    /**
     * 添加一条记录
     */
    add(record) {
        const list = this.getAll();
        list.unshift({
            id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6),
            timestamp: new Date().toISOString(),
            ...record,
        });
        // 保留最多 HISTORY_MAX 条
        if (list.length > HISTORY_MAX) {
            list.length = HISTORY_MAX;
        }
        try {
            localStorage.setItem(HISTORY_KEY, JSON.stringify(list));
        } catch {
            // localStorage 满了，清理一半
            list.length = Math.floor(HISTORY_MAX / 2);
            localStorage.setItem(HISTORY_KEY, JSON.stringify(list));
        }
        return list;
    },

    /**
     * 清空所有记录
     */
    clear() {
        localStorage.removeItem(HISTORY_KEY);
    },

    /**
     * 渲染历史记录到 DOM
     */
    render(containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const list = this.getAll();
        if (list.length === 0) {
            container.innerHTML = '<div class="history-empty">暂无记录</div>';
            return;
        }

        container.innerHTML = "";
        list.forEach((item) => {
            const el = document.createElement("div");
            el.className = "history-item";
            el.title = `${item.class_name} ${(item.confidence * 100).toFixed(1)}%`;

            const color = CLASS_COLORS[item.class_name] || "#6B7280";
            el.innerHTML = `
                <img class="history-thumb" src="${item.thumbnail}" alt="${item.class_name}">
                <div class="history-class" style="color:${color}">${item.class_name}</div>
                <div class="history-conf">${(item.confidence * 100).toFixed(0)}%</div>
            `;

            el.addEventListener("click", () => {
                // 点击历史记录，触发自定义事件
                window.dispatchEvent(new CustomEvent("history-select", { detail: item }));
            });

            container.appendChild(el);
        });
    },
};
