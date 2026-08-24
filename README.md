# DeepSeek-V4-Flash-0731 on RTX PRO 6000 Blackwell

无容器、`uv` 本地编译部署 DeepSeek-V4-Flash-0731。默认使用 4 张 RTX PRO
6000 Blackwell（GPU 0-3），开启固定 probabilistic DSpark K5，服务监听
`127.0.0.1:8000`。

## 实测吞吐

短上下文基准：128 input tokens、256 output tokens、`temperature=0`、
`ignore_eos=true`。这是本机 TP4/DCP1/1M 配置的实测值，不是长上下文或质量承诺。

| 并发 | 请求数 | Output tok/s | Total tok/s | 结果 |
|---:|---:|---:|---:|---:|
| C1 | 4 | **212.53** | 318.80 | 4/4 |
| C4 | 16 | **287.06** | 430.59 | 16/16 |
| C8 | 32 | **389.59** | 584.39 | 32/32 |

原始结果见 [`benchmarks/`](benchmarks/)。显式发送 `temperature=0` 对本配置很重要；
同一 C1 测试不发送该字段时为 131.34 output tok/s。

## 配置亮点

- 模型：官方 `deepseek-ai/DeepSeek-V4-Flash-0731`，权重目录默认是
  `/data1/models/DeepSeek-V4-Flash-0731`；
- 推理栈：Infernal Invocation r18、B12X W4A8、FP8 compressed MLA KV、DSpark K5；
- 上下文：`MAX_MODEL_LEN=1048576`（1M），默认调度槽位 16；
- 接口：OpenAI 兼容 API，默认 reasoning，思考放在独立 `message.reasoning`；
- 工具调用：启用 DeepSeek V4 reasoning/tool-call parser，输出标准 `tool_calls[]`；
- 服务使用 GPU 0-3（4 卡）。

上游 r18 的合格收据是 TP2/DCP1；本文记录的是在同一源码基线上验证过的本机 TP4/1M
扩展配置。

## 快速部署

### 1. 前置条件

Linux、CUDA 13.3、Python 3.12、`uv`、`git`、`ninja`、`cmake`、`numactl`，以及至少
约 350 GiB 的模型、源码和编译缓存空间。确认 GPU：

```bash
nvidia-smi -L
nvidia-smi topo -m
```

网络需要代理时，显式传入：

```bash
export PROXY_URL='http://user:password@proxy-host:port'
```

### 2. 准备权重

权重已存在时跳过。否则任选 Hugging Face 或 ModelScope：

```bash
mkdir -p /data1/models
uvx --from huggingface_hub hf download \
  deepseek-ai/DeepSeek-V4-Flash-0731 \
  --local-dir /data1/models/DeepSeek-V4-Flash-0731
```

```bash
uvx --from modelscope modelscope download \
  --model deepseek-ai/DeepSeek-V4-Flash-0731 \
  --local_dir /data1/models/DeepSeek-V4-Flash-0731
```

### 3. 编译 uv 环境

```bash
git clone https://github.com/wildoranges/deepseek-v4-flash-rtxpro6k.git
cd deepseek-v4-flash-rtxpro6k
./setup-infernal-r18.sh
```

脚本会在当前目录创建 `.venv-infernal-r18`，拉取并固定 r18 所需源码，编译
FlashInfer、XGrammar、B12X 和 vLLM 的 Blackwell 扩展。首次编译较久，缓存默认放在
`/data1/uv-cache`；可用 `MAX_JOBS=16 NVCC_THREADS=4` 降低并行度。

### 4. 启动服务

```bash
umask 077
openssl rand -hex 32 > .api_key
./serve-infernal-r18.sh
```

生产运行建议放入 tmux：

```bash
tmux new-session -d -s ds_r18_prod \
  './serve-infernal-r18.sh >> server-infernal-r18-8000.log 2>&1'
```

```bash
curl -i http://127.0.0.1:8000/health
```

## 默认参数

| 参数 | 默认值 |
|---|---:|
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3` |
| `TP_SIZE` / `DCP_SIZE` | `4` / `1` |
| `MAX_MODEL_LEN` | `1048576` |
| `MAX_NUM_SEQS` | `16` |
| `MAX_NUM_BATCHED_TOKENS` | `4096` |
| `GPU_MEMORY_UTILIZATION` | `0.975` |
| `DSPARK_TOKENS` / `DSPARK_DEPTH_MODE` | `5` / `fixed` |
| `OMP_NUM_THREADS` / `NUMA_NODE` | `16` / `0` |
| `PORT` / `HOST` | `8000` / `127.0.0.1` |

例如改端口或显式指定 key：

```bash
PORT=8001 VLLM_API_KEY='your-key' ./serve-infernal-r18.sh
```

## API 调用

```bash
API_KEY="$(tr -d '\r\n' < .api_key)"
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer ${API_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "dsv4-flash",
    "messages": [{"role": "user", "content": "用一句话解释 speculative decoding"}],
    "temperature": 0,
    "max_tokens": 256
  }'
```

默认开启 reasoning，解析结果位于 `message.reasoning`，正文位于
`message.content`。单次请求关闭思考：

```json
{"chat_template_kwargs":{"thinking":false}}
```

可按请求设置思考强度：

```json
{"chat_template_kwargs":{"thinking":true,"reasoning_effort":"high"}}
```

支持值由模型模板决定；本部署验证过 `low`、`medium`、`high`、`max`，服务默认 `max`。

## 1M 上下文与缓存

当前 KV 预算约 7,339,750 tokens，按 1,048,576-token 满上下文请求估算理论上限约 7
路；这不是 7 路满上下文的性能保证。真实请求越长，decode 吞吐越低。

服务端已启用 prefix cache，复用 vLLM 的 KV block；可查看 vLLM metrics 或响应中的
`prompt_tokens_details.cached_tokens`。

## 常见排查

- 工具调用出现在正文：确认使用仓库脚本启动，并检查日志中是否启用了
  `--tokenizer-mode deepseek_v4`、`--tool-call-parser deepseek_v4`、
  `--reasoning-parser deepseek_v4`、`--enable-auto-tool-choice`。
- 队列有请求但 GPU 利用率低：先排除 stale 请求和 JIT，结合生成 tok/s、功耗和
  target-step rate 判断，不要只看 1 秒粒度的 SM 百分比。
- 重启后首次请求慢：不要删除 `/data1/uv-cache/vllm-dsv4-infernal-r18`，其中包含
  CUDA Graph 和 B12X JIT 缓存。

## 上游资料

- [RTX PRO 6000 Blackwell / SM120 部署资料](https://github.com/local-inference-lab/rtx6kpro)
- [Infernal Invocation r18 DS4 配方](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/ds4dspark-infernal-invocation-r18.md)
- [DeepSeek-V4-Flash-0731 模型](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)

本仓库只包含无容器部署脚本、已验证配置和基准结果；模型权重及第三方源码按其各自许可
使用。自有脚本与文档采用 [MIT License](LICENSE)。
