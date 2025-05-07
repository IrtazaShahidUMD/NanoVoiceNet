# NanoVoiceNet 🎤⚡️
**Low-Latency Transformer for Real-Time Speech Enhancement**

NanoVoiceNet is a compact, Transformer-based, low-latency speech enhancement model that denoises raw audio in real-time. It operates on 64-sample chunks, making it ideal for streaming and low-latency audio applications such as VoIP, hearing aids, and embedded devices.

---

## 🧠 Model Highlights

- **Raw Audio Input**: Processes 1D waveforms directly in the time domain.
- **Chunk-Wise Processing**: Operates on 64-sample chunks with context from past audio.
- **Transformer-Based**: Leverages multi-head self-attention for temporal modeling.
- **Lightweight Design**: Optimized for fast inference and edge deployment.
- **Improves SNR**: Trained to denoise using a clean-noisy speech pair dataset.

---

## 🛠️ Architecture

- **Encoder**: 1D Convolutional layer projecting raw audio to 32-D latent space
- **Positional Embeddings**: Captures temporal structure across chunks
- **Transformer Encoder**: 4 layers, 2 heads, 64 FFN dim
- **Decoder**: Linear layer projecting back to raw audio space
- **Loss Function**: MSE between predicted and clean waveforms

---

## 🚀 Training

- Dataset: 1000 clean-noisy speech file pairs
- Optimizer: Adam
- Loss: MSE
- Notes: Validation loss was still decreasing at end of training but stopped early due to compute constraints

---

## 📦 Usage

```bash
git clone https://github.com/yourusername/NanoVoiceNet.git
cd NanoVoiceNet
