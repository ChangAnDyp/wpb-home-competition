#!/usr/bin/env bash
# 启动本机 Ollama 服务（动作识别用的视觉模型跑在它上面）
#
# 用法：bash ~/competition/start_ollama.sh
#
# 说明：
#   - ollama 安装在 ~/.local，没有走系统包管理，所以要显式指定路径。
#   - 推理本身不需要联网；下面的代理变量只在**下载模型**时才用得到，
#     而且必须配合 Clash 里选中的“中转/IEPL”节点（“直连”节点会慢 15 倍）。
#   - 模型：qwen3-vl:2b-instruct（2.1B，支持图片输入）

set -u

OLLAMA_BIN="$HOME/.local/bin/ollama"
CLASH_PROXY="http://127.0.0.1:7897"

if [ ! -x "$OLLAMA_BIN" ]; then
    echo "找不到 $OLLAMA_BIN，Ollama 没装好。" >&2
    exit 1
fi

if pgrep -x ollama >/dev/null; then
    echo "Ollama 已在运行（pid $(pgrep -x ollama | head -1)）。"
else
    echo "启动 Ollama ..."
    setsid nohup env \
        PATH="$HOME/.local/bin:/usr/bin:/bin" \
        OLLAMA_HOST=127.0.0.1:11434 \
        HTTPS_PROXY="$CLASH_PROXY" \
        HTTP_PROXY="$CLASH_PROXY" \
        NO_PROXY=127.0.0.1,localhost \
        "$OLLAMA_BIN" serve > /tmp/ollama_serve.log 2>&1 &
    sleep 6
fi

echo -n "服务版本: "
curl -sS --max-time 5 http://127.0.0.1:11434/api/version || {
    echo "启动失败，看 /tmp/ollama_serve.log" >&2
    exit 1
}
echo
echo "已安装的模型："
"$OLLAMA_BIN" list
