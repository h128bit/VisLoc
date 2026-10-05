from pathlib import Path
from datasets.AnyVisLoc import AnyVisLocDataset

root = Path(r"C:\Users\t.asmus\Desktop\data\AnyVisLoc\Scene_10\Scene_10")

satellite_resize = 1024
image_resize = 512
p = [root]

data = AnyVisLocDataset(p, satellite_resize, image_resize)

train = data.get_train_part()
test = data.get_test_part() 

point, sat, image = train[0]

print(point)
print(sat.shape)
print(image.shape)