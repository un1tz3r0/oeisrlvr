# Gemma 4 - How to Run Locally

Gemma 4 is Google DeepMind’s new family of open models, including **12B**, **E2B**, **E4B**, **26B-A4B**, and **31B.** The multimodal, hybrid-thinking models support 140+ languages, up to **256K context**, and have dense and MoE variants. Gemma 4 is Apache-2.0 licensed and can run on your local device.

**Gemma-4-12B** is new and features unified text, image and audio support. It runs on **8GB** RAM (4-bit) or 14GB (8-bit). **Gemma-4-E2B** and **E4B** also support image and audio. Run on **5GB RAM** (4-bit) or 15GB (full 16-bit).

<a href="/pages/VnmWq1kNppQrTqCI6aLH#run-gemma-4-tutorials" class="button primary">Run Gemma 4</a><a href="/pages/6iXghkDoe3jzknTq5aWx" class="button secondary">Fine-tune Gemma 4</a><a href="/pages/9kjF1F7Gsb0dgOA9lcpl" class="button primary">Gemma 4 QAT</a><a href="/pages/3PWlU172DOGeqxIflfP7#gemma-4-mtp" class="button secondary">Gemma 4 MTP</a>

{% hint style="success" %}
**NEW:** [**Gemma 4 MTP is here**](/docs/models/mtp.md)**! MTP enables 1.4-2.2x faster inference without accuracy loss. Run MTP directly in** [**Unsloth Studio**](/docs/models/mtp.md#unsloth-studio-mtp-guide)**.**
{% endhint %}

{% columns %}
{% column %}
**Gemma-4-26B-A4B** runs on **18GB** (4-bit) or 28GB (8-bit). **Gemma-4-31B** needs **20GB RAM** (4-bit) or 34GB (8-bit).

You can now run all GGUFs, [MLX](#mlx-dynamic-quants) and fine-tune Gemma 4 in [Unsloth Studio](#unsloth-studio-guide) (see right).

[**QAT** variants](/docs/models/gemma-4/qat.md) of Gemma 4 reduce memory requirements around 3x while preserving model quality.
{% endcolumn %}

{% column %}

<div data-with-frame="true"><figure><img src="/files/3rrq58PcvZFnywcYbnPb" alt=""><figcaption></figcaption></figure></div>
{% endcolumn %}
{% endcolumns %}

{% hint style="success" %}
**Jun 9:** [Gemma 4 MTP](/docs/models/mtp.md) is here.

**Jun 5:** [Gemma 4 QAT](/docs/models/gemma-4/qat.md) is released.

**Jun 2:** Gemma 4 12B Unified is released.

**Apr 20:** We conducted [Gemma 4 GGUF Benchmarks](#unsloth-gguf-benchmarks) to help you pick the best quant.
{% endhint %}

### Usage Guide

Gemma 4 excels at reasoning, coding, tool use, long-context and agentic workflows, and multimodal tasks. The smaller E2B and E4B variants are designed for phones and laptops, while the larger models target medium-high CPU /VRAM systems such as PCs with NVIDIA RTX GPUs.

| Gemma 4 Variant | Details                                                          | Best fit                                              |
| --------------- | ---------------------------------------------------------------- | ----------------------------------------------------- |
| **E2B**         | <p>Dense + PLE (128K context)<br>Support: Text, Image, Audio</p> | For phone / edge inference, ASR, speech translation   |
| **E4B**         | <p>Dense + PLE (128K context)<br>Support: Text, Image, Audio</p> | Small model for laptops and fast local multimodal use |
| **12B Unified** | <p>Dense (256K context)<br>Support: Text, Image, Audio</p>       | Medium model for laptops and local multimodal use     |
| **26B-A4B**     | <p>MoE (256K context)<br>Support: Text, Image</p>                | Best speed / quality tradeoff for computer use        |
| **31B**         | <p>Dense (256K context)<br>Support: Text, Image</p>              | Strongest performance at slower inference             |

**See Gemma 4:** [**Performance benchmarks**](#official-gemma-benchmarks) **and** [**GGUF benchmarks**](#unsloth-gguf-benchmarks)**.**

**Should I pick 26B-A4B or 31B?**

* **26B-A4B** - balances speed and accuracy. Its MoE design makes it faster than 31B, with 4B active parameters. Pick it if RAM is limited and you are fine trading a bit of quality for speed.
* **31B** - currently the strongest Gemma 4 model. Pick it for maximum quality if you have enough memory and can accept slightly slower speeds.

### Hardware requirements

**Table: Gemma 4 Inference GGUF recommended hardware requirements** (units = total memory: RAM + VRAM, or unified memory). You can use Gemma 4 on MacOS, NVIDIA RTX GPUs etc.

| Gemma 4 variant |    4-bit |    8-bit | BF16 / FP16 |
| --------------- | -------: | -------: | ----------: |
| **E2B**         |     4 GB |   5–8 GB |       10 GB |
| **E4B**         | 5.5–6 GB |  9–12 GB |       16 GB |
| **12B Unified** |   7–8 GB | 13–14 GB |       25 GB |
| **26B A4B**     | 16–18 GB | 28–30 GB |       52 GB |
| **31B**         | 17–20 GB | 34–38 GB |       62 GB |

{% hint style="info" %}
As a rule of thumb, your total available memory should at least exceed the size of the quantized model you download. If it does not, llama.cpp can still run using partial RAM / disk offload, but generation will be slower. You will also need more compute, depending on the context window you use.
{% endhint %}

### Recommended Settings

It is recommended to use Google's default Gemma 4 parameters:

* `temperature = 1.0`
* `top_p = 0.95`
* `top_k = 64`

{% hint style="info" %}
Gemma 4's max context is **128K** for **E2B** / **E4B** and `262,144` for **12B** / **26B A4B** / **31B**.
{% endhint %}

#### Thinking Mode

Compared to older Gemma chat templates, Gemma 4 uses the standard **`system`**, **`assistant`**, and **`user`** roles and adds explicit thinking control.

**How to enable thinking:**

Add the token **`<|think|>`** at the **start of the system prompt**.

{% columns %}
{% column %}
**Thinking enabled**

```
<|think|>
You are a careful coding assistant. Explain your answer clearly.
```

{% endcolumn %}

{% column %}
**Thinking disabled**

```
You are a careful coding assistant. Explain your answer clearly.
```

{% endcolumn %}
{% endcolumns %}

**Output behavior:**

{% columns %}
{% column %}
When thinking is enabled, the model outputs its internal reasoning channel before the final answer.

```
<|channel>thought
[internal reasoning]
<channel|>
[final answer]
```

{% endcolumn %}

{% column %}
When thinking is disabled, the larger models may still emit an **empty thought block** before the final answer.

```
<|channel>thought
<channel|>
[final answer]
```

{% endcolumn %}
{% endcolumns %}

**For example using "**&#x57;hat is the capital of France?":

{% code overflow="wrap" %}

```
<bos><|turn>system\n<|think|><turn|>\n<|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n
```

{% endcode %}

**then it outputs with:**

{% code overflow="wrap" %}

```
<|channel>thought\nThe user is asking for the capital of France.\nThe capital of France is Paris.<channel|>The capital of France is Paris.<turn|>
```

{% endcode %}

**Multi-turn chat rule:**

For multi-turn conversations, **only keep the final visible answer in chat history**. Do **not** feed prior thought blocks back into the next turn.

{% code overflow="wrap" %}

```
<bos><|turn>user\nWhat is 1+1?<turn|>\n<|turn>model\n2<turn|>\n<|turn>user\nWhat is 1+1?<turn|>\n<|turn>model\n2<turn|>\n<|turn>user\nWhat is 1+1?<turn|>\n<|turn>model\n2<turn|>\n<|turn>user\nWhat is 1+1?<turn|>\n<|turn>model\n2<turn|>\n
```

{% endcode %}

**How to disable thinking:**

Note `llama-cli` might not work reliably, so use `llama-server` for disabling reasoning:

{% hint style="warning" %}
To [disable thinking / reasoning](#how-to-enable-or-disable-reasoning-and-thinking), use `--chat-template-kwargs '{"enable_thinking":false}'`

If you're on **Windows** Powershell, use: `--chat-template-kwargs "{\"enable_thinking\":false}"`

Use 'true' and 'false' interchangeably.
{% endhint %}

## Run Gemma 4 Tutorials

Because Gemma 4 GGUFs comes in several sizes, the recommended starting point for the small models is 8-bit and the larger models is [**Dynamic**](/docs/basics/unsloth-dynamic-2.0-ggufs.md) **4-bit**. [Gemma 4 GGUFs](https://huggingface.co/collections/unsloth/gemma-4) or [MLX](#mlx-dynamic-quants):

| [E2B](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF) | [E4B](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF) | [12b](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF) | [26B-A4B](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) | [31B](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF) |
| --------------------------------------------------------- | --------------------------------------------------------- | --------------------------------------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------- |

<a href="/pages/VnmWq1kNppQrTqCI6aLH#unsloth-studio-guide" class="button primary">🦥 Unsloth Studio Guide</a><a href="/pages/VnmWq1kNppQrTqCI6aLH#llama.cpp-guide" class="button primary">🦙 Llama.cpp Guide</a>

{% columns %}
{% column %}
**You can run and train Gemma 4 for free with a UI in our** [**Unsloth Studio**](/docs/new/studio.md)✨ **notebook:**
{% endcolumn %}

{% column %}
{% embed url="<https://colab.research.google.com/github/unslothai/unsloth/blob/main/studio/Unsloth_Studio_Colab.ipynb>" %}
{% endcolumn %}
{% endcolumns %}

### 🦥 Unsloth Studio Guide

Gemma 4 can now be run and fine-tuned in [Unsloth Studio](/docs/new/studio.md), our new open-source web UI for local AI. Unsloth Studio lets you run models locally on **MacOS, Windows**, Linux and:

{% columns %}
{% column %}

* Search, download, [run GGUFs](/docs/new/studio.md#run-models-locally) and safetensor models
* [**Self-healing** tool calling](/docs/new/studio.md#execute-code--heal-tool-calling) + **web search**
* [**Code execution**](/docs/new/studio.md#run-models-locally) (Python, Bash)
* [Automatic inference](/docs/new/studio.md#model-arena) parameter tuning (temp, top-p, etc.)
* Fast CPU + GPU inference via llama.cpp
* [Train LLMs](/docs/new/studio.md#no-code-training) 2x faster with 70% less VRAM
  {% endcolumn %}

{% column %}

<div data-with-frame="true"><figure><img src="/files/XNsT8Jn9t1xo3KuFpe4G" alt=""><figcaption></figcaption></figure></div>
{% endcolumn %}
{% endcolumns %}

{% stepper %}
{% step %}

#### Install Unsloth

Run in your terminal:

**MacOS, Linux, WSL:**

```bash
curl -fsSL https://unsloth.ai/install.sh | sh
```

**Windows PowerShell:**

```bash
irm https://unsloth.ai/install.ps1 | iex
```

{% endstep %}

{% step %}

#### Launch Unsloth

**MacOS, Linux, WSL and Windows:**

```bash
unsloth studio -H 0.0.0.0 -p 8888
```

Then open `http://127.0.0.1:8888` in your browser.
{% endstep %}

{% step %}

#### Search and download Gemma 4

On first launch you will need to create a password to secure your account and sign in again.

Then go to the [Studio Chat](/docs/new/studio/chat.md) tab and search for Gemma 4 in the search bar and download your desired model and quant. Unsloth supports the latest Gemma-4-12B Unified model.

<div data-with-frame="true"><figure><img src="/files/BZBZ2cFeu1ibTtMH84tv" alt="" width="375"><figcaption></figcaption></figure></div>
{% endstep %}

{% step %}

#### Run Gemma 4

Inference parameters should be auto-set when using Unsloth Studio, however you can still change it manually. You can also edit the context length, chat template and other settings. You can run GGUFs and MLX files.

For more information, you can view our [Unsloth Studio inference guide](/docs/new/studio/chat.md).

<div data-with-frame="true"><figure><img src="/files/XNsT8Jn9t1xo3KuFpe4G" alt="" width="563"><figcaption></figcaption></figure></div>
{% endstep %}
{% endstepper %}

### 🦙 Llama.cpp Guide

For this guide we will be utilizing Dynamic 4-bit for the 12B, 26B-A4B and 31B, and 8-bit for E2B and E4B.  See: [Gemma 4 GGUF collection](https://huggingface.co/collections/unsloth/gemma-4)

For these tutorials, we will using [llama.cpp](llama.cpphttps://github.com/ggml-org/llama.cpp) for fast local inference, especially if you have a CPU.

{% stepper %}
{% step %}
Obtain the latest `llama.cpp` **on** [**GitHub here**](https://github.com/ggml-org/llama.cpp). You can follow the build instructions below as well. Change `-DGGML_CUDA=ON` to `-DGGML_CUDA=OFF` if you don't have a GPU or just want CPU inference. **For Apple Mac / Metal devices**, set `-DGGML_CUDA=OFF` then continue as usual - Metal support is on by default.

```bash
apt-get update
apt-get install pciutils build-essential cmake curl libcurl4-openssl-dev -y
git clone https://github.com/ggml-org/llama.cpp
cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON
cmake --build llama.cpp/build --config Release -j --clean-first --target llama-cli llama-mtmd-cli llama-server llama-gguf-split
cp llama.cpp/build/bin/llama-* llama.cpp
```

{% endstep %}

{% step %}
If you want to use `llama.cpp` directly to load models, you can follow commands below, according to each model. `UD-Q4_K_XL` is the quantization type. You can also download via Hugging Face (step 3). This is similar to `ollama run` . Use `export LLAMA_CACHE="folder"` to force `llama.cpp` to save to a specific location. There is no need to set context length as llama.cpp automatically uses the exact amount required.

{% hint style="warning" %}
To [disable thinking / reasoning](#how-to-enable-or-disable-reasoning-and-thinking), use: `--chat-template-kwargs '{"enable_thinking":false}'`

**Windows** Powershell: `--chat-template-kwargs "{\"enable_thinking\":false}"`

Use '`true`' and '`false`' interchangeably.
{% endhint %}

**12B:**

```bash
export LLAMA_CACHE="unsloth/gemma-4-12B-it-GGUF"
./llama.cpp/llama-cli \
    -hf unsloth/gemma-4-12b-it-GGUF:UD-Q4_K_XL \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 64
```

**26B-A4B:**

```bash
export LLAMA_CACHE="unsloth/gemma-4-26B-A4B-it-GGUF"
./llama.cpp/llama-cli \
    -hf unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 64
```

**31B:**

```bash
export LLAMA_CACHE="unsloth/gemma-4-31B-it-GGUF"
./llama.cpp/llama-cli \
    -hf unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 64
```

**E4B:**

```bash
export LLAMA_CACHE="unsloth/gemma-4-E4B-it-GGUF"
./llama.cpp/llama-cli \
    -hf unsloth/gemma-4-E4B-it-GGUF:Q8_0 \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 64
```

**E2B:**

```bash
export LLAMA_CACHE="unsloth/gemma-4-E2B-it-GGUF"
./llama.cpp/llama-cli \
    -hf unsloth/gemma-4-E2B-it-GGUF:Q8_0 \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 64
```

{% endstep %}

{% step %}
You can also download the model manually as well via the code below (after installing `pip install huggingface_hub`). You can choose `UD-Q4_K_XL` or other quantized versions like `Q8_0` . If downloads get stuck, see: [Hugging Face Hub, XET debugging](/docs/basics/troubleshooting-and-faqs/hugging-face-hub-xet-debugging.md)

```bash
hf download unsloth/gemma-4-26B-A4B-it-GGUF \
    --local-dir unsloth/gemma-4-26B-A4B-it-GGUF \
    --include "*mmproj-BF16*" \
    --include "*UD-Q4_K_XL*" # Use "*UD-Q2_K_XL*" for Dynamic 2bit
```

{% endstep %}

{% step %}
Then run the model in conversation mode (with vision `mmproj-F16`):

{% code overflow="wrap" %}

```bash
./llama.cpp/llama-cli \
    --model unsloth/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf \
    --mmproj unsloth/gemma-4-26B-A4B-it-GGUF/mmproj-BF16.gguf \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 64
```

{% endcode %}
{% endstep %}

{% step %}

#### Llama-server deployment

To deploy Gemma-4 on llama-server, use:

```bash
./llama.cpp/llama-server \
    --model unsloth/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf \
    --mmproj unsloth/gemma-4-26B-A4B-it-GGUF/mmproj-BF16.gguf \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 64 \
    --alias "unsloth/gemma-4-26B-A4B-it-GGUF" \
    --port 8001 \
    --chat-template-kwargs '{"enable_thinking":true}'
```

{% endstep %}
{% endstepper %}

### MLX Dynamic Quants

We also uploaded dynamic 4bit and 8bit quants as a first trial for MacOS device! The MLX quants support **vision.**

{% hint style="success" %}
All MLX quants now work in[ Unsloth Studio](#unsloth-studio-guide)!
{% endhint %}

| Gemma 4 | 4-bit MLX                                                             | 8-bit MLX                                                          |
| ------- | --------------------------------------------------------------------- | ------------------------------------------------------------------ |
| 31B     | [link](https://huggingface.co/unsloth/gemma-4-31b-it-UD-MLX-4bit)     | [link](https://huggingface.co/unsloth/gemma-4-31b-it-MLX-8bit)     |
| 26B-A4B | [link](https://huggingface.co/unsloth/gemma-4-26b-a4b-it-UD-MLX-4bit) | [link](https://huggingface.co/unsloth/gemma-4-26b-a4b-it-MLX-8bit) |
| E4B     | [link](https://huggingface.co/unsloth/gemma-4-E4B-it-UD-MLX-4bit)     | [link](https://huggingface.co/unsloth/gemma-4-E4B-it-MLX-8bit)     |
| E2B     | [link](https://huggingface.co/unsloth/gemma-4-E2B-it-UD-MLX-4bit)     | [link](https://huggingface.co/unsloth/gemma-4-E2B-it-MLX-8bit)     |

To try them out use:

{% code overflow="wrap" %}

```bash
curl -fsSL https://raw.githubusercontent.com/unslothai/unsloth/refs/heads/main/scripts/install_gemma4_mlx.sh | sh
source ~/.unsloth/unsloth_gemma4_mlx/bin/activate
python -m mlx_vlm.chat --model unsloth/gemma-4-26b-a4b-it-UD-MLX-4bit
```

{% endcode %}

### Ollama Guide

Ollama now supports Unsloth GGUFs well now. Use `curl -fsSL https://ollama.com/install.sh | sh` to install Ollama on Linux or `irm https://ollama.com/install.ps1 | iex` for Windows.\
\
To use a single quant file (under 50GB) use:

{% code overflow="wrap" %}

```bash
ollama run hf.co/unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL
```

{% endcode %}

For multiple shards like larger BF16 shards do:

{% code overflow="wrap" %}

```bash
pip install -U huggingface_hub

# Download mmproj and BF16 in 2 calls
hf download unsloth/gemma-4-26B-A4B-it-GGUF --include "BF16/*" \
    --local-dir gemma4
hf download unsloth/gemma-4-26B-A4B-it-GGUF --include "mmproj-BF16.gguf" \
    --local-dir gemma4

mv gemma4/mmproj-BF16.gguf gemma4/BF16/
echo "FROM ./gemma4/BF16" > Modelfile

ollama create unsloth-gemma4 -f Modelfile
ollama run unsloth-gemma4
```

{% endcode %}

<div data-with-frame="true"><figure><img src="/files/QLViFbJkmFCoUO4zFWQA" alt="" width="563"><figcaption></figcaption></figure></div>

{% hint style="info" %}
If you see `Error: 500 Internal Server Error: unable to load model` update Ollama via `curl -fsSL https://ollama.com/install.sh | sh` or use the Powershell one.
{% endhint %}

## Gemma 4 Best Practices

### Prompting examples

#### Simple reasoning prompt

```
System:
<|think|>
You are a precise reasoning assistant.

User:
A train leaves at 8:15 AM and arrives at 11:47 AM. How long was the journey?
```

#### OCR / document prompt

For OCR, use a **high visual token budget** like **560** or **1120**.

```
[image first]
Extract all text from this receipt. Return line items, total, merchant, and date as JSON.
```

#### Multi-modal comparison prompt

```
[image 1]
[image 2]
Compare these two screenshots and tell me which one is more likely to confuse a new user.
```

#### Audio ASR prompt

```
Transcribe the following speech segment in {LANGUAGE} into {LANGUAGE} text.

Follow these specific instructions for formatting the answer:
* Only output the transcription, with no newlines.
* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three.
```

#### Audio translation prompt

```
Transcribe the following speech segment in {SOURCE_LANGUAGE}, then translate it into {TARGET_LANGUAGE}. When formatting the answer, first output the transcription in {SOURCE_LANGUAGE}, then one newline, then output the string '{TARGET_LANGUAGE}: ', then the translation in {TARGET_LANGUAGE}.
```

### Multi-modal Settings

For best results with multimodal prompts, put multimodal content first:

* Put **image and/or audio before text**.
* For video, pass a sequence of frames first, then the instruction.

#### Audio and video limits

* **Audio** is available on **12B**, **E2B** and **E4B** only.
* Audio supports a maximum of **30 seconds**.
* Video supports a maximum of **60 seconds** assuming **1 frame per second** processing.

#### Audio prompt templates

**ASR prompt**

```
Transcribe the following speech segment in {LANGUAGE} into {LANGUAGE} text.

Follow these specific instructions for formatting the answer:
* Only output the transcription, with no newlines.
* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three.
```

**Speech translation prompt**

```
Transcribe the following speech segment in {SOURCE_LANGUAGE}, then translate it into {TARGET_LANGUAGE}.
When formatting the answer, first output the transcription in {SOURCE_LANGUAGE}, then one newline, then output the string '{TARGET_LANGUAGE}: ', then the translation in {TARGET_LANGUAGE}.
```

## 📊 Benchmarks

### Unsloth GGUF Benchmarks

We conducted Mean KL Divergence benchmarks for Gemma 4 GGUFs across providers to help you pick the best quant (lower is better).

* KL Divergence puts all Unsloth GGUFs on the SOTA Pareto frontier
* KLD shows how well a quantized model matches the original BF16 output distribution, indicating retained accuracy.

<div data-with-frame="true"><figure><img src="/files/t4dOLPTn4r4eZjngFoTc" alt=""><figcaption><p>26B A4B - KLD benchmarks (lower is better)</p></figcaption></figure></div>

### Official Gemma Benchmarks

**Text/Code Benchmarks**

| Benchmark           | Gemma 4 31B | Gemma 4 26B A4B | Gemma 4 12B Unified | Gemma 4 E4B | Gemma 4 E2B | Gemma 3 27B (no think) |
| ------------------- | ----------- | --------------- | ------------------- | ----------- | ----------- | ---------------------- |
| MMLU Pro            | 85.2%       | 82.6%           | 77.2%               | 69.4%       | 60.0%       | 67.6%                  |
| AIME 2026 no tools  | 89.2%       | 88.3%           | 77.5%               | 42.5%       | 37.5%       | 20.8%                  |
| LiveCodeBench v6    | 80.0%       | 77.1%           | 72.0%               | 52.0%       | 44.0%       | 29.1%                  |
| Codeforces ELO      | 2150        | 1718            | 1659                | 940         | 633         | 110                    |
| GPQA Diamond        | 84.3%       | 82.3%           | 78.8%               | 58.6%       | 43.4%       | 42.4%                  |
| Tau2                | 76.9%       | 68.2%           | 69.0%               | 42.2%       | 24.5%       | 16.2%                  |
| HLE no tools        | 19.5%       | 8.7%            | 5.2%                | -           | -           | -                      |
| HLE with search     | 26.5%       | 17.2%           | -                   | -           | -           | -                      |
| BigBench Extra Hard | 74.4%       | 64.8%           | 53.0%               | 33.1%       | 21.9%       | 19.3%                  |
| MMMLU               | 88.4%       | 86.3%           | 83.4%               | 76.6%       | 67.4%       | 70.7%                  |

**Vision Benchmarks**

| MMMU Pro                           | 76.9% | 73.8% | 69.1% | 52.6% | 44.2% | 49.7% |
| ---------------------------------- | ----- | ----- | ----- | ----- | ----- | ----- |
| OmniDocBench 1.5 (lower is better) | 0.131 | 0.149 | 0.164 | 0.181 | 0.290 | 0.365 |
| MATH-Vision                        | 85.6% | 82.4% | 79.7% | 59.5% | 52.4% | 46.0% |
| MedXPertQA MM                      | 61.3% | 58.1% | 48.7% | 28.7% | 23.5% | -     |

**Audio Benchmarks**

| CoVoST                          | -     | -     | 38.5<sup>\*</sup>  | 35.54 | 33.47 | -     |
| ------------------------------- | ----- | ----- | ------------------ | ----- | ----- | ----- |
| FLEURS (lower is better)        | -     | -     | 0.069<sup>\*</sup> | 0.08  | 0.09  | -     |
| **Long Context**                |       |       |                    |       |       |       |
| MRCR v2 8 needle 128k (average) | 66.4% | 44.1% | 43.4%              | 25.4% | 19.1% | 13.5% |

<div data-with-frame="true"><figure><img src="/files/VEt5TNCRPr3cjfW7McPC" alt=""><figcaption></figcaption></figure></div>
