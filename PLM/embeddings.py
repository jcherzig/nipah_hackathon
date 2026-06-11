######################################### Pytorch imports
##################################################
import torch
from torch.utils.data import DataLoader
######################################### Transformers imports
##############################################
# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM
######################################### HuggingFace Hub imports
##############################################
# ESM-C weights are gated — run `huggingface-cli login` once, or call login() below
from huggingface_hub import login
######################################### Numpy and Pandas imports
############################################
import numpy as np
import pandas as pd
import json
import re
######################################### tqdm imports
##################################################
from tqdm import tqdm
from Bio import SeqIO

############################################################################################################
######################################## Input parameters
##########################################################
params = {
    #Select model
    # Available ESM-C checkpoints: biohub/ESMC-300M  biohub/ESMC-600M  biohub/ESMC-6B
    'model_name' : "facebook/esm2_t33_650M_UR50D",  # ESM-2 (no login needed)
    # 'model_name' : "biohub/ESMC-600M",
}

############################################################################################################
######################################## Global Variables
##########################################################
#Amino acid tokens
amino_acids = ["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]

############################################################################################################
######################################## Load Model into GPU
#######################################################
# ESM-C weights are gated on HuggingFace. Authenticate before first download:
#   huggingface-cli login    (run once in terminal)
# Or uncomment and pass your token directly:
# login(token="hf_...")
login()

#Load model and tokenizer
# Note: for ESM-C, output_hidden_states is passed at inference time, not here
model = AutoModelForMaskedLM.from_pretrained(params['model_name'])
tokenizer = AutoTokenizer.from_pretrained(params['model_name'])

#Assign device (MPS for Apple Silicon, CUDA if available, otherwise CPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

model = model.to(device)
model.eval()

############################################################################################################
######################################## embed_batch (mirrors HuggingFace_Functions.py)
##########################################################
def embed_batch(sequences, tokenizer, model, device):
    """Return (log_softmax logits, mean embeddings, token ids) for a batch of sequences."""
    batch_lsoftmax = torch.nn.LogSoftmax(dim=2)

    with torch.no_grad():
        tokenized = tokenizer(sequences, return_tensors="pt").to(device)
        sequence_length = len(sequences[0])

        output = model(**tokenized, output_hidden_states=True)

        # Logits: strip <cls> and <eos>
        logits = output.logits[:, 1:sequence_length+1, :].to('cpu')
        logits = np.array(batch_lsoftmax(logits))

        # Mean embedding from final hidden layer, strip <cls> and <eos>
        embeddings = np.array([
            e.mean(axis=0)
            for e in output.hidden_states[-1][:, 1:sequence_length+1, :].to('cpu')
        ])

        token_ids = tokenized.to('cpu')['input_ids'][:, 1:-1]

    return np.array(logits), np.array(embeddings), np.array(token_ids)

############################################################################################################
######################################## Main
##########################################################
if __name__ == "__main__":
    fasta_file = 'all_sequences.fasta'  # Path to your FASTA file
    batch_size = 1  # Adjust based on your GPU memory


    sequences_and_embeddings = {}
    with open(fasta_file) as input_handle:
        for record in SeqIO.parse(input_handle, "fasta"):
            sequences_and_embeddings[record.id] = {}
            logits, embeddings, token_ids = embed_batch([str(record.seq)], tokenizer, model, device)
            logits,embeddings = logits[0],embeddings[0]
            sequences_and_embeddings[record.id]['sequence'] = str(record.seq)
            sequences_and_embeddings[record.id]['logits'] = sequence_logits = logits[np.arange(logits.shape[0]), token_ids].tolist()
            sequences_and_embeddings[record.id]['embeddings'] = embeddings.tolist()
    # Save the results to a JSON file
    with open('all_sequences_and_embeddings.json', 'w') as output_handle:
        json.dump(sequences_and_embeddings, output_handle)
        
