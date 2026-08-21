# Corrected-evaluation statistics

## fiqa

| model | n | acc | wF1 | macroF1 | unparse | McNemar p (vs Finistral) | Δacc 95% CI |
|---|---|---|---|---|---|---|---|
| Finistral-7B-LoRA | 235 | 0.8766 | 0.8833 | 0.6999 | 0.0000 |  |  |
| FinGPT-Falcon-7B-LoRA **<-- strongest baseline** | 235 | 0.8553 | 0.8594 | 0.7226 | 0.0000 | 0.4237 | [-0.0213, +0.0638] |
| FinGPT-mt-Llama2-7B-LoRA | 235 | 0.8085 | 0.8301 | 0.6743 | 0.0000 | 0.01242 | [+0.0213, +0.1191] |
| Finistral-7B-LoRA (Alpaca prompt) | 235 | 0.7745 | 0.8209 | 0.6525 | 0.0000 | 0.0001907 | [+0.0553, +0.1532] |
| FinGPT-Llama-3-8B-LoRA | 235 | 0.6213 | 0.7219 | 0.5795 | 0.0723 | 7.247e-11 | [+0.1872, +0.3234] |
| FinGPT-Bloom-7B1-LoRA | 235 | 0.6936 | 0.6914 | 0.5283 | 0.0000 | 2.88e-07 | [+0.1191, +0.2468] |
| FinBERT (ProsusAI) | 235 | 0.5149 | 0.6147 | 0.4726 | 0.0000 | 3.111e-17 | [+0.2936, +0.4298] |
| Mistral-7B-v0.1-Base | 235 | 0.0128 | 0.0252 | 0.0176 | 0.9872 | 1.26e-45 | [+0.8170, +0.9064] |

Template ablation: [INST] 0.8766 vs Alpaca 0.7745 (Δ = +0.1021); see the Alpaca row above for McNemar/CI vs the [INST] row.

## fpb_decontam

| model | n | acc | wF1 | macroF1 | unparse | McNemar p (vs Finistral) | Δacc 95% CI |
|---|---|---|---|---|---|---|---|
| Finistral-7B-LoRA | 560 | 0.9893 | 0.9893 | 0.9853 | 0.0000 |  |  |
| FinGPT-mt-Llama2-7B-LoRA **<-- strongest baseline** | 560 | 0.9857 | 0.9857 | 0.9798 | 0.0000 | 0.7744 | [-0.0089, +0.0161] |
| Finistral-7B-LoRA (Alpaca prompt) | 560 | 0.9804 | 0.9804 | 0.9755 | 0.0000 | 0.1797 | [-0.0018, +0.0196] |
| FinGPT-Falcon-7B-LoRA | 560 | 0.9696 | 0.9697 | 0.9633 | 0.0000 | 0.007385 | [+0.0071, +0.0339] |
| FinBERT (ProsusAI) | 560 | 0.9643 | 0.9648 | 0.9518 | 0.0000 | 0.006611 | [+0.0089, +0.0429] |
| FinGPT-Llama-3-8B-LoRA | 560 | 0.9339 | 0.9410 | 0.9265 | 0.0179 | 8.14e-07 | [+0.0357, +0.0768] |
| FinGPT-Bloom-7B1-LoRA | 560 | 0.8946 | 0.8934 | 0.8517 | 0.0000 | 5.701e-11 | [+0.0679, +0.1214] |
| Mistral-7B-v0.1-Base | 560 | 0.0036 | 0.0070 | 0.0135 | 0.9946 | 1.258e-121 | [+0.9750, +0.9946] |

Template ablation: [INST] 0.9893 vs Alpaca 0.9804 (Δ = +0.0089); see the Alpaca row above for McNemar/CI vs the [INST] row.

## tfns

| model | n | acc | wF1 | macroF1 | unparse | McNemar p (vs Finistral) | Δacc 95% CI |
|---|---|---|---|---|---|---|---|
| Finistral-7B-LoRA (Alpaca prompt) | 2373 | 0.8786 | 0.8811 | 0.8598 | 0.0000 | 1.675e-33 | [-0.1037, -0.0754] |
| FinGPT-Llama-3-8B-LoRA **<-- strongest baseline** | 2373 | 0.7804 | 0.8211 | 0.7906 | 0.0927 | 0.4067 | [-0.0110, +0.0287] |
| Finistral-7B-LoRA | 2373 | 0.7893 | 0.7959 | 0.7776 | 0.0000 |  |  |
| FinGPT-mt-Llama2-7B-LoRA | 2373 | 0.7644 | 0.7713 | 0.7527 | 0.0000 | 0.002143 | [+0.0097, +0.0405] |
| FinGPT-Falcon-7B-LoRA | 2373 | 0.7307 | 0.7377 | 0.7223 | 0.0000 | 2.417e-11 | [+0.0417, +0.0754] |
| FinBERT (ProsusAI) | 2373 | 0.7252 | 0.7329 | 0.6679 | 0.0000 | 9.459e-09 | [+0.0426, +0.0860] |
| FinGPT-Bloom-7B1-LoRA | 2373 | 0.6827 | 0.6986 | 0.6449 | 0.0000 | 1.19e-21 | [+0.0851, +0.1281] |
| Mistral-7B-v0.1-Base | 2373 | 0.0013 | 0.0025 | 0.0047 | 0.9945 | 0 | [+0.7716, +0.8041] |

Template ablation: [INST] 0.7893 vs Alpaca 0.8786 (Δ = -0.0893); see the Alpaca row above for McNemar/CI vs the [INST] row.
