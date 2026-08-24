# 在 RTX PRO 6000 Blackwell 上部署 DeepSeek-V4-Flash-0731

这是一份不用容器、通过 `uv` 在本机编译并部署
[`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
的实战教程。参考实现来自
[`local-inference-lab/rtx6kpro`](https://github.com/local-inference-lab/rtx6kpro)
的 Infernal Invocation r18 配方。

本仓库的默认目标是：

- 4 张 **NVIDIA RTX PRO 6000 Blackwell Workstation Edition**；
- GPU 0-3，TP4，只使用一组 4 卡；
- 官方 DeepSeek-V4-Flash-0731 权重；
- B12X W4A8、FP8 compressed MLA KV；
- fixed probabilistic DSpark K5，不关闭 DSpark；
- OpenAI 兼容 API，监听 `127.0.0.1:8000`；
- 最大模型上下文 `1,048,576` tokens；
- 默认 reasoning，并把思考内容解析到独立字段；
- 标准 OpenAI `tool_calls[]`，避免把 `<tool_call>` 当普通正文返回。

> [!NOTE]
> 本文实测配置为 TP4/DCP1、DSpark K5 和 1M 模型上下文上限；上游 r18 的参考
> 配置为 TP2/DCP1。性能数据均来自本文所列的本机环境。

## 测试环境

| 项目 | 实测配置 |
|---|---|
| 主机 GPU | 8× NVIDIA RTX PRO 6000 Blackwell Workstation Edition |
| 单卡显存 | 驱动可见 97,887 MiB |
| 服务使用 | 4× GPU（GPU 0-3），TP4，DCP1 |
| 操作系统 | Ubuntu 26.04 LTS |
| NVIDIA 驱动 | 595.71.05 |
| CUDA Toolkit | 13.3 |
| Python | 3.12 |
| uv | 0.12.5 |
| vLLM | `0.26.1rc0+infernal.invocation...r18` |
| 模型磁盘大小 | 约 155.43 GiB |

版本不是宽松建议，而是这套源码组合的已验证基线。特别是 CUDA、PyTorch、
FlashInfer、B12X 和 vLLM 之间耦合较强，升级其中一项前应重新做工具调用、
正确性与吞吐测试。

## 准备系统

需要 Linux、CUDA 13.3、足够的内存和磁盘空间。模型、源码、编译目录和缓存放在
一起时，建议至少预留 350 GiB，500 GiB 更从容。

Ubuntu 可先安装基础工具：

```bash
sudo apt update
sudo apt install -y build-essential cmake curl git ninja-build numactl pkg-config tmux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

确认 GPU、拓扑和 CUDA：

```bash
nvidia-smi -L
nvidia-smi topo -m
/usr/local/cuda-13.3/bin/nvcc --version
```

如果网络需要代理，显式传入自己的代理地址：

```bash
export PROXY_URL='http://user:password@proxy-host:port'
```

## 下载模型

权重默认放在 `/data1/models/DeepSeek-V4-Flash-0731`。任选一种来源：

```bash
sudo mkdir -p /data1/models
sudo chown "${USER}:${USER}" /data1/models

uvx --from huggingface_hub hf download \
  deepseek-ai/DeepSeek-V4-Flash-0731 \
  --local-dir /data1/models/DeepSeek-V4-Flash-0731
```

或使用 ModelScope：

```bash
uvx --from modelscope modelscope download \
  --model deepseek-ai/DeepSeek-V4-Flash-0731 \
  --local_dir /data1/models/DeepSeek-V4-Flash-0731
```

模型页面：

- <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731>
- <https://modelscope.ai/models/deepseek-ai/DeepSeek-V4-Flash-0731>

## 构建 uv 环境

```bash
git clone https://github.com/wildoranges/rtx-pro-6000-blackwell-deepseek-v4-flash.git
cd rtx-pro-6000-blackwell-deepseek-v4-flash
./setup-infernal-r18.sh
```

脚本会：

1. 创建 `.venv-infernal-r18`；
2. 拉取固定提交的 vLLM、B12X、FlashInfer、XGrammar 和 r18 补丁；
3. 校验组合后的 vLLM/B12X Git tree；
4. 针对 SM120a 编译 CUDA、FlashInfer 和 vLLM 扩展；
5. 安装并核对精确版本。

默认构建缓存位于 `/data1/uv-cache`。可按机器内存调整并行度：

```bash
MAX_JOBS=16 NVCC_THREADS=4 ./setup-infernal-r18.sh
```

这是完整源码构建，冷启动耗时较长。不要因为几分钟没有新输出就中断编译。

## 创建 API Key 并启动

```bash
umask 077
openssl rand -hex 32 > .api_key
./serve-infernal-r18.sh
```

推荐在 tmux 中运行：

```bash
tmux new-session -d -s ds_r18_prod \
  './serve-infernal-r18.sh >> server-infernal-r18-8000.log 2>&1'
```

健康检查：

```bash
curl -i http://127.0.0.1:8000/health
```

默认启动参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3` | 使用前 4 张卡 |
| `TP_SIZE` | `4` | Tensor Parallel 4 |
| `DCP_SIZE` | `1` | DSpark 当前要求 DCP1 |
| `MAX_MODEL_LEN` | `1048576` | 1M 模型上限 |
| `MAX_NUM_SEQS` | `16` | 调度器并发槽位 |
| `MAX_NUM_BATCHED_TOKENS` | `4096` | chunked prefill 上限 |
| `GPU_MEMORY_UTILIZATION` | `0.975` | KV/图内存预算 |
| `DSPARK_TOKENS` | `5` | fixed K5 |
| `DSPARK_DEPTH_MODE` | `fixed` | 已验证的固定深度 |
| `OMP_NUM_THREADS` | `16` | r18 上游默认值 |
| `NUMA_NODE` | `0` | 默认匹配本机 GPU 0-3 |

GPU 4-7 在本测试机属于 NUMA 1，测试第二组卡时使用：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 NUMA_NODE=1 PORT=8001 \
  ./serve-infernal-r18.sh
```

不同主板拓扑可能不同，请以 `nvidia-smi topo -m` 和 `numactl --hardware` 为准。

## 调用 API

普通请求：

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

默认开启 reasoning，解析后的思考位于 `message.reasoning`，不会故意混入
`message.content`。单次请求关闭思考：

```json
{
  "chat_template_kwargs": {
    "thinking": false
  }
}
```

推理强度可传 `reasoning_effort`：

```json
{
  "chat_template_kwargs": {
    "thinking": true,
    "reasoning_effort": "high"
  }
}
```

支持值取决于模型模板；本部署使用过 `low`、`medium`、`high` 和 `max`，服务默认
为 `max`。

## New API 接入

在 New API 中使用：

- Base URL：`http://127.0.0.1:8000/v1`
- 模型：`dsv4-flash`
- Key：`.api_key` 中的值

关闭 `-nothink` 别名思考时，可以在 New API 请求转换中设置：

```json
{
  "operations": [
    {
      "mode": "set",
      "path": "chat_template_kwargs.thinking",
      "value": false,
      "conditions": [
        {
          "path": "model",
          "mode": "suffix",
          "value": "-nothink"
        }
      ],
      "logic": "OR"
    }
  ]
}
```

vLLM prefix cache 是服务端 KV block cache。New API 不一定把它显示成供应商缓存计费
字段；应查看 vLLM metrics、日志中的 `Prefix cache hit rate`，或响应
`prompt_tokens_details.cached_tokens`。

## 1M 上下文与并发

本机 TP4/1M/`MAX_NUM_SEQS=16` 启动后分配了约 `7,339,750` 个 KV tokens。
仅按 KV 容量估算，满 1,048,576-token 请求的理论并发约为：

```text
7,339,750 / 1,048,576 ≈ 7.00
```

这不是 7 路满上下文性能保证。还需考虑输出增长、调度水位、激活内存和请求形状。
实际部署应保留余量。

更重要的是，`MAX_MODEL_LEN=1M` 本身不会让短请求变慢；真正影响 decode 的是请求
实际携带的上下文。一次生产观察中，6 个请求平均约 155K 上下文时，总生成吞吐
只有约 18-25 tok/s，而短上下文基准能超过 200 tok/s。

## 实测吞吐

测试参数为 128-token 输入、256-token 输出、`temperature=0`、`ignore_eos=true`。

| 最大并发 | 请求数 | output tok/s | total tok/s | 成功率 |
|---:|---:|---:|---:|---:|
| C1 | 4 | 212.53 | 318.80 | 4/4 |
| C4 | 16 | 287.06 | 430.59 | 16/16 |
| C8 | 32 | 389.59 | 584.39 | 32/32 |

原始 JSON 位于 [`benchmarks/`](benchmarks/)。这些数字适合验证部署是否明显退化，
不代表长上下文或真实 agent 工作负载的速度。

### temperature 对 DSpark 的影响

同一 C1 基准不显式发送 `temperature` 时，output throughput 只有 `131.34 tok/s`，
DSpark 平均接受长度为 `2.00`；显式使用 `temperature=0` 时为 `212.53 tok/s`，
平均接受长度为 `4.30`。

如果业务允许确定性解码，显式设置 `temperature=0` 通常更利于这套 DSpark 配置。
不要只为了速度改变需要随机采样的业务语义。

## 部署经验

### 1. 不要把显存占满误认为算力跑满

本部署每卡显存约 96.4-96.5/97.9 GiB，但长上下文请求下 SM 利用率会间歇性归零，
功耗也可能只有 80-100W。原因通常是长上下文稀疏索引、TP4 PCIe 同步和小 decode
batch 之间的空洞，不是模型没有加载。

### 2. DSpark 与 DCP

r18 的 DeepSeek V4 DSpark 非因果注意力要求 `DCP_SIZE=1`。本教程不通过关闭
DSpark 换取 DCP4；所有默认配置都保留 fixed probabilistic DSpark K5。

### 3. NUMA 很重要

优化前 EngineCore 曾落在远端 NUMA 节点。把 API、EngineCore、worker 和内存绑定到
GPU 0-3 所在的 NUMA 0 后，C4 的 raw output throughput 略有提升；按 DSpark 接受
长度归一化后的 target-step 速度提升约 10%。

### 4. 提高槽位不等于提高实际并发

`MAX_NUM_SEQS=16` 只是上限。若 metrics 显示 Running=6、Waiting=0，继续增加服务端
队列不会填满 GPU；需要让调用方实际发送更多独立请求。并发增大会提高总吞吐，
但通常牺牲单请求延迟。

### 5. Prefix cache 只省 prefill

高 prefix-cache 命中率不会消除长上下文每一步 decode 的稀疏索引与注意力成本。
Agent/Harness 应对旧历史做摘要或裁剪，保留 64K-128K 活跃窗口通常比盲目维持
数百 K 上下文更有效。

### 6. 首次启动很慢是正常现象

155 GiB 权重需要数分钟加载，之后还要做 CUDA Graph 内存分析、图捕获和可能的
B12X JIT。JIT 缓存默认保存在 `/data1/uv-cache/vllm-dsv4-infernal-r18`，不要在每次
重启前删除。

## 常见问题

### 工具调用为什么出现在普通正文里？

典型异常是响应 `finish_reason: stop`，正文含 `<tool_call>...</tool_call>`，但没有
OpenAI `tool_calls`。本部署显式启用：

```text
--tokenizer-mode deepseek_v4
--tool-call-parser deepseek_v4
--reasoning-parser deepseek_v4
--enable-auto-tool-choice
```

更新 vLLM、模板或模型版本后必须重新验证普通、streaming 和并发工具调用。

### 队列满但 GPU 利用率为什么为 0？

先检查请求是否 stale、EngineCore 是否仍有进展、KV 是否 preempt，以及日志是否在
JIT。对于长上下文 decode，1 秒粒度的 `nvidia-smi` 也可能错过短促 kernel burst。
结合生成 tok/s、功耗、SM、显存带宽和 target-step rate 判断，不要只看一个百分比。

### 为什么接入 New API 后看不到缓存？

这是 vLLM 本地 prefix cache，不是上游厂商的计费缓存协议。缓存可以有效，但 New API
前端不一定显示对应字段。

## 安全说明

- `.api_key`、日志、权重、缓存、虚拟环境和源码构建树已被仓库白名单排除；
- 服务默认只监听 localhost；
- 不要把带用户名/密码的代理 URL 写入脚本或提交历史；
- 如需对外暴露，请额外配置 TLS、访问控制、限流和反向代理。

## 上游资料与致谢

- [RTX PRO 6000 Blackwell 部署资料](https://github.com/local-inference-lab/rtx6kpro)
- [Infernal Invocation r18 DS4 配方](https://github.com/local-inference-lab/rtx6kpro/blob/main/models/ds4dspark-infernal-invocation-r18.md)
- [r18 源码合并契约](https://github.com/local-inference-lab/rtx6kpro/issues/67)
- [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)

感谢 local-inference-lab 对 RTX PRO 6000 Blackwell 本地推理栈的公开研究。本仓库只把
经过本机验证的无容器步骤、配置和排障经验整理为可复现教程。

## License

本仓库自有脚本与文档使用 [MIT License](LICENSE)。下载和构建的第三方项目、补丁与
模型权重分别受各自许可证约束。
