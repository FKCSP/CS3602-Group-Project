# CS3602-Group-Project
The final project of CS3602(natural language processing---SJTU IEEE Honor Class).

## Environment Setup
    conda create -n slu python=3.8
    pytorch installation
    conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
    - CPU version: pip install torch==1.7.1
    - GPU version: pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html

## File Explanation
- `utils/args.py`: definitions of all related optional parameters
- `utils/initialization.py`: Initialize system settings, including setting random seed and GPU/CPU
- `utils/vocab.py`: Build a vocabulary for encoding input and output
- `utils/word2vec.py`: Load word vector
- `utils/example.py`: Read data
- `utils/batch.py`: Load input data as batches
- `model/slu_baseline_tagging.py`: Baseline model
- `scripts/slu_baseline.py`: Main program script
