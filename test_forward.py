import timm 
import torch 

test_hid = 80

print("CREATE resnet18")
resnet18_emb = timm.create_model("resnet18", pretrained=False)
resnet18 = timm.create_model("resnet18", pretrained=False)

resnet18_emb.fc = torch.nn.Linear(in_features=512, out_features=test_hid, bias=True)
resnet18.fc = torch.nn.Linear(in_features=512, out_features=2, bias=True)

from ghn3 import GHN3 

max_shape = (1000, 1000, 10, 10)
num_clasees = 2
hid = test_hid

print("CREATE GHN")
ghn = GHN3(max_shape, num_clasees, hid)


image = torch.rand(3, 224, 224).unsqueeze(0)
emb = resnet18_emb(image)

arch = [resnet18]
res = ghn(arch, emb)

print(type(res[0]))