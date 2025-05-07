import os
import numpy as np
import argparse
import soundfile as sf


from speech_enhancement_model import Model, ModelInferenceWrapper


def initialize():
    """
    Initializes the model, loads weights, and resets states
    """
    
    return ModelInferenceWrapper()

    
def snr(est: np.ndarray, gt: np.ndarray, EPS=1e-9):
    """
    Calculates the signal-to-noise ratio (SNR)
    est: Estimate, or model output
    gt: Ground truth clean signal
    """
    s_pwr = np.sum(gt**2) + EPS
    n_pwr = np.sum((gt-est)**2) + EPS
    
    snr = 10 * np.log10(s_pwr / n_pwr)
    return snr


def main(args):
    dset = args.dataset

    snr_os = []
    snr_streaming = []
    os_streaming_maxdiff = [] # max absolute error between one-shot and streaming model
    
    for sample_name in os.listdir(dset):
        sample_dir = os.path.join(dset, sample_name)

        # Load mixture audio
        mix_path = os.path.join(sample_dir, 'mix.wav')
        mix, sr = sf.read(mix_path)
        assert sr == 16000

        # Load ground truth (gt) audio
        gt_path = os.path.join(sample_dir, 'gt.wav')
        gt, sr = sf.read(gt_path)
        assert sr == 16000
        
        # Initialize one-shot and streaming models (make sure you correctly load the same weights)
        model_streaming = initialize()

        
        # Run inference on 4ms audio chunks
        # Split audio into 4 ms chunks
        streaming_chunks = np.split(mix, indices_or_sections=mix.shape[-1] // 64) # Split audio into 4 ms chunks (64 samples)
        out_streaming = []
        for chunk in streaming_chunks:
            out_chunk = model_streaming(chunk)
            out_streaming.append(out_chunk)
        out_streaming = np.concatenate(out_streaming, axis=0)

        gt = gt[:out_streaming.shape[-1]]

        # [Evaluate]
        snr_streaming.append(snr(out_streaming, gt))
        
        print("gt:", gt.shape)
        print("mix:", mix.shape)
        print("out_os:", out_os.shape)
        print("chunk:", chunk.shape)
        print("out_chunk:", out_chunk.shape)
        
        #break
    
    
    print(f'Streaming SNR: {np.mean(snr_streaming)}dB')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='sample_dataset/',
                        help='Path to dataset')
    
    main(parser.parse_args())
