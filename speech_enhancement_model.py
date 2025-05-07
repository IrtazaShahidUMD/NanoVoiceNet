import torch
import torch.nn as nn
import numpy as np


class Model(nn.Module):
    """
    Speech enhancement model using:
    - Conv1D encoder
    - Transformer encoder for temporal modeling
    - Conv1D decoder for waveform reconstruction
    The model processes input in fixed-size chunks with stateful memory across chunks.
    """
    
    def __init__(self):
        super(Model, self).__init__()
        
        # Model hyperparameters
        kernel_size = 5
        d_model = 32 
        n_head = 2 
        num_layers = 4
        dim_feedforward = 64
        chunk_size = 64
        num_past_chunks = 2
        
        # Save configuration
        self.kernel_size = kernel_size
        self.chunk_size = chunk_size
        self.num_past_chunks = num_past_chunks
        self.d_model = d_model
        
        # Encoder: transforms raw audio into feature embeddings
        self.encoder = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=kernel_size)
        self.act = nn.ReLU()
        
        # Positional embeddings for temporal order
        self.positional_embedding_table = nn.Embedding((num_past_chunks+1)*chunk_size, d_model)
        
        # Transformer encoder for modeling temporal dependencies
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Decoder: projects back to 1D waveform space
        self.decoder = nn.Conv1d(in_channels=d_model, out_channels=1, kernel_size=kernel_size)
        
        
        
    def initialize_states(self, batch_size, device):
        """
        Initializes internal states used for chunk-based streaming inference.

        Args:
            batch_size (int): Batch size of the current input.
            device (str): Device to allocate the state tensors to.

        Returns:
            dict: A dictionary of internal state tensors.
        """
        
        return {
            'past_state':torch.zeros((batch_size, 1, self.kernel_size-1)).to(device),
            'past_conv':torch.zeros((batch_size, self.d_model, self.chunk_size*self.num_past_chunks)).to(device),
            'transformer_state':torch.zeros((batch_size, self.d_model, self.kernel_size - 1)).to(device)
        }

    def forward(self, audio, states = None, device='cpu'):
        """
        Forward pass of the model on a chunk of audio.

        Args:
            audio (Tensor): Input tensor of shape [B, 1, T] where T == chunk_size.
            states (dict, optional): Internal states passed across chunks.
            device (str): Device to ensure consistency in state creation.

        Returns:
            Tuple[Tensor, dict]: Output audio tensor and updated internal states.
        """
        
        # Initialize states if not provided
        if states is None:
            states = self.initialize_states(audio.shape[0], device)
        
        
        # Use and update past state
        audio = torch.cat([states['past_state'], audio], dim=-1) # Concatenate previous state (time-context)
        states['past_state'] = audio[..., -states['past_state'].shape[-1]:] # Update state for next chunks
        
        x = self.encoder(audio) #  Output: [B, d_model, T]
        x = self.act(x)         #  Output: [B, d_model, T]
        
        x = torch.cat([states['past_conv'], x], dim=-1) # Concatenate previous state (time-context)
        states['past_conv'] = x[..., -states['past_conv'].shape[-1]:] # Update state for next chunks
        
        # x at this instance [B, d_model, (chunk_size + 1) * T]
        
        x = x = x.transpose(1, 2) # Output: [B, (chunk_size + 1) * T, d_model]
        
        # generating positional embedding
        pos = self.positional_embedding_table(torch.arange((self.num_past_chunks+1)*self.chunk_size, device=device))
        
        x = x + pos # Output: [B, (chunk_size + 1) * T, d_model]
        
        x = self.transformer(x) #  Output: [B, (chunk_size + 1) * T, d_model]
        
        x = x = x.transpose(1, 2) #  Output: [B, d_model, (chunk_size + 1) * T]
        
        x = x[..., -self.chunk_size:] #  Output: [B, d_model, T]
        
        x = torch.cat([states['transformer_state'], x], dim=-1) # Concatenate previous state (time-context)
        states['transformer_state'] = x[..., -states['transformer_state'].shape[-1]:] # Update state for next chunks
        
        #  x at this instance [B, d_model, T+Kernel_size-1]
        
        out = self.decoder(x) #  Output: [B, 1, T]
        
        return out, states


class ModelInferenceWrapper(object):
    """
    Wrapper class for real-time inference using the speech enhancement model.
    Maintains internal state across streaming chunks.
    """
    
    def __init__(self):
        # Initialize model
        self.nn = Model()
        self.nn.eval()

        # Load model and set to evaluation mode
        self.nn.load_state_dict(torch.load('speech_enhancement_model.pt'))

        # Initialize internal state for streaming inference
        self.states = self.nn.initialize_states(1, device = 'cpu')

    def __call__(self, audio):
        """
        Enhances a chunk of input audio.

        Args:
            audio (np.ndarray): Input waveform of shape [T,].

        Returns:
            np.ndarray: Enhanced waveform of shape [T,].
        """
        
        with torch.inference_mode():
            audio = torch.from_numpy(audio).reshape(1,1,-1).float() # Add batch & channel dims

            output, self.states = self.nn(audio, self.states)

            output = output.reshape(-1).numpy()

        return output

if __name__ == "__main__":
    model = Model()
    torch.save(model.state_dict(), "my_model_weights.pt")