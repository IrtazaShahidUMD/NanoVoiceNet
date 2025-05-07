# NanoVoiceNet 🎤⚡️
**Low-Latency Transformer for Real-Time Speech Enhancement**

NanoVoiceNet is a compact, Transformer-based, low-latency speech enhancement model that denoises raw audio in real-time. It operates on 64-sample chunks, making it ideal for streaming and low-latency audio applications such as VoIP, hearing aids, and embedded devices.

---

## 🧠 Model Highlights

- **Raw Audio Input**: Processes 1D waveforms directly in the time domain
- **Streaming-Ready**: Enhances audio chunk-by-chunk with minimal latency
- **Context-Aware**: Incorporates multiple past chunks for more stable and coherent speech enhancement
- **Transformer-Based**: Leverages self-attention for temporal modeling
- **Lightweight Design**: Optimized for fast inference and on-device deployment
- **Trained Weights Provided**: Pretrained model weights are included for quick evaluation and testing


---

## 🛠️ Architecture

- **Encoder**: 1D Convolutional layer projecting raw audio to 32-D latent space
- **Positional Embeddings**: Captures temporal structure across chunks
- **Transformer Encoder**: 4 layers, 2 heads, 64 FFN dim
- **Decoder**: Linear layer projecting back to raw audio space
- **Loss Function**: L1 loss between predicted and clean waveforms

> The model is designed to process current chunks along with a configurable number of past chunks (e.g., 2), enabling **temporal continuity** and smoother audio output.
---

## 🚀 Training

- Dataset: 2 hours of clean-noisy speech file pairs
- Optimizer: AdamW
- Loss: L1 Loss
- Notes: Validation loss was still decreasing at the end of training, but stopped early due to computational constraints
- Status: ✅ Pretrained model weights are provided
---

## 📦 Usage

```bash
git clone https://github.com/yourusername/NanoVoiceNet.git
cd NanoVoiceNet
```

## 🔭 Future Work
- Integrate perceptual loss (e.g., PESQ, SI-SNR) for human-quality improvements
- Quantization & knowledge distillation for further edge optimization
- Scale to larger, more diverse datasets
- Resume training to continue improving performance

---

Developed by [Irtaza Shahid](www.cs.umd.edu/~irtaza/)

Feedback, suggestions, and contributions are welcome!
