import os
if __name__ == '__main__' and __package__ is None:
    import sys
    # use_encoder.py の1階層上（プロジェクトのルートディレクトリ）を sys.path に追加
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # カレントディレクトリを変更
    os.chdir(target_dir)
    __package__ = 'scripts'

from .infer import InferClass
import torch
from torchviz import make_dot
def get_embed(input):
    infer_class = InferClass(config_file = os.path.join(os.path.dirname(__file__), '..', "configs/infer.yaml"))
    embed = infer_class.model.image_encoder.encoder(input.to(infer_class.device))
    return embed[-1]

def check_infer():
    infer_class = InferClass(config_file = os.path.join(os.path.dirname(__file__), '..', "configs/infer.yaml"))
    print(infer_class.model.image_encoder) 
    input = torch.rand(1,1,128,128,128).to(infer_class.device)
    output = infer_class.model.image_encoder(input)
    embed = infer_class.model.image_encoder.encoder(input)
    print(f"{infer_class.model=}")
    for i, emb in enumerate(embed):
        print(f"{emb.shape=}, {i=}")
    #image = make_dot(output, params=dict(infer_class.model.image_encoder.named_parameters()))
    #image.format = "png"
    #image.render("NeuralNet")
    #print(len(output), output[0].shape, output[1].shape)
    #print(f"{len(embed)=}, {embed[0].shape=}")
    #model_scripted = torch.jit.trace(infer_class.model.image_encoder, input) # Export to TorchScript
    #model_scripted.save('model_scripted2.pt') # Save

def main():
    input = torch.rand(1,1,128,128,128)
    embed = get_embed(input)
    print(embed.shape)
    
if __name__ == "__main__":
    main()