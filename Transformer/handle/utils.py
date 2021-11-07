import torch


def handle_device(args):
    if args.device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            UserWarning("No Cuda detected. Running on cpu.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    return device
