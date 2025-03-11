import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

from modelutils import *
from quant import Quantizer
from gptq import GPTQ

def get_model_and_input_output(args_dataset, args_nsamples, args_seed, args_model, inps_path="", outs_path=""):
    # Load a pre-trained GPT-2 medium model (hidden size = 1024)
    model = GPT2LMHeadModel.from_pretrained(args_model)
    model.transformer.h = model.transformer.h[:1]
    model.eval()

    # Select a layer with a weight matrix of shape (1024, 1024)
    # Here we use the projection layer in the first transformer block's attention module.
    layer = model.transformer.h[0].attn.c_proj
    layer.eval()  # set to evaluation mode
    print("Original weight shape:", layer.weight.shape)  # Expected: torch.Size([1024, 1024])

    if inps_path != "" and outs_path != "":
        inps = torch.load(inps_path)
        outs = torch.load(outs_path)
        return (layer, inps, outs)

    # ----- Step 3. Generate a list of sample inputs and collect calibration data -----
    seqlen = 1024
    dataloader, testloader = get_loaders(
        args_dataset, nsamples=args_nsamples, seed=args_seed, model=args_model, seqlen=seqlen
    )

    in_size = (args_nsamples, layer.weight.shape[1], seqlen)
    out_size = (args_nsamples, layer.weight.shape[0], seqlen)
    inps = []
    outs = []
    
    def add_batch():
        def tmp(_, inp, out):
            inps.append(inp[0].data)
            outs.append(out.data)
            #gptq_obj.add_batch(inp[0].data, out.data)
        return tmp

    handles = []
    handles.append(layer.register_forward_hook(add_batch()))
    
    #data gathering
    with torch.no_grad():
        i = 0
        for batch in dataloader:
            print(i, batch[0])
            model(batch[0])
            i+=1
            
    #delete forward hooks
    for h in handles:
        print(h)
        h.remove()
    print("end")
    
    inps = torch.stack(inps)
    outs = torch.stack(outs)
    
    #get input and output
    return (layer, inps, outs)

if __name__ == '__main__':
    from datautils import *

    args_dataset = "wikitext2"
    args_nsamples = 128
    args_seed = 0
    args_model = "gpt2-medium"

    layer, inps, outs = get_model_and_input_output(args_dataset, args_nsamples, args_seed, args_model, inps_path="data/inps.pt", outs_path="data/outs.pt")

    #torch.save(inps, "data/inps.pt")
    #torch.save(outs, "data/outs.pt")

    print(layer)
    print(inps.shape)
    print(outs.shape)
    
    
    # Save a copy of the original weights (optional, for later loss computation)
    original_weight = layer.weight.data.clone()

    # ----- Step 2. Set up the GPTQ quantizer for this layer -----
    # Create a GPTQ object for the target layer.
    gptq_obj = GPTQ(layer)
    # Instantiate and configure the quantizer for 4-bit quantization.
    gptq_obj.quantizer = Quantizer()
    gptq_obj.quantizer.configure(bits=4, perchannel=True, sym=False, mse=False)
    
    # add batch to gptq
    for i in range(inps.shape[0]):
        gptq_obj.add_batch(inps[i], outs[i])

    # ----- Step 4. Quantize the layer using fasterquant() -----
    # Here we call fasterquant() to quantize the layer’s weights to 4-bit.
    quant_error = gptq_obj.fasterquant(
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
        static_groups=False
    )

    # If fasterquant() does not return a value, you could compute the mean squared error
    # between the original weights and the quantized weights as a proxy:
    if quant_error is None:
        quant_error = torch.nn.functional.mse_loss(layer.weight.data, original_weight).item()

    print("Quantization Loss (Error):", quant_error)
    