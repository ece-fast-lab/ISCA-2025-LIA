# Policy Search (Prefill/Decode Plots)

This script produces **two side-by-side plots** showing which policy is optimal across a grid of batch size (`B`) and sequence length (`L`) using a latency model for OPT-family Transformers under different CPU/GPU placement **policies**.

- **Left plot:** **Prefill**
- **Right plot:** **Decode**

Each point `(B, L)` is colored by the best (lowest latency) policy among all 64 binary vectors of length 6.

---
## How to run (Command-line arguments)

This script accepts **CPU type**, **GPU type**, and **model name** as command-line inputs.  
This is the **recommended** way to run the code for reproducibility and automation.

### Command format
```bash
python policy_search.py --cpu <CPU_TYPE> --gpu <GPU_TYPE> --model <MODEL_NAME>
```

### Supported inputs

#### CPU_TYPE
```text
SPR   # Sapphire Rapids
GNR   # Granite Rapids
```

#### GPU_TYPE
```text
A100
H100
```

#### Model_TYPE
```text
OPT-30B   # d_model = 7168
OPT-66B   # d_model = 9216
OPT-175B  # d_model = 12288
```
