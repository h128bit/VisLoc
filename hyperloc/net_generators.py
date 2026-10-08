import timm 

def get_model(name, emb_size):
    return timm.create_model(name, num_classes=emb_size, pretrained=False)


def generate_resnet18(number: int):
    def generate():
        return [get_model("resnet18", emb_size=2) for _ in range(number)]

    return generate