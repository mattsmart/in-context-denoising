import torch
#import torch_directml
#dml = torch_directml.device()


def device_select():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = "cpu"
    print('device selected:', device)
    return device


if __name__ == '__main__':
    """
    Windows machine -- Use DirectML with AMD GPU
    """
    try:
        import torch_directml
        dml_device = torch_directml.device()
        print('Direct ML device loaded.')
        #tensor1 = torch.tensor([1]).to(dml_device)  # Note that dml is a variable, not a string!
        x = torch.ones(1, device=dml_device)
        print('Check Direct ML:', x)
    except ImportError as err:
        print(err)
        dml_device = None

    """
    Macbook with M1 chip
    """
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print('\nCheck M1 chip:', x)
    else:
        print('\nMPS device not found.')
        mps_device = None

    """
    Generic case (cpu)
    """
    cpu_device = 'cpu'
    x = torch.ones(1, device=cpu_device)
    print('\nCheck cpu:', x)
