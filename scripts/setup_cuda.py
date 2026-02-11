#!/usr/bin/env python3
"""
Setup script per PyTorch CUDA su RTX 5080/5090 Blackwell (sm_120).

Uso:
    uv run --no-sync python scripts/setup_cuda.py

Questo script:
1. Verifica se torch CUDA nightly è installato
2. Se no, lo installa automaticamente
3. Verifica che la GPU sia riconosciuta
"""
import subprocess
import sys


def check_cuda():
    """Verifica se CUDA è disponibile."""
    try:
        import torch
        return torch.cuda.is_available(), torch.__version__
    except ImportError:
        return False, None


def install_torch_nightly():
    """Installa torch nightly con CUDA 12.8."""
    print("Installing PyTorch nightly with CUDA 12.8...")
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--index-url", "https://download.pytorch.org/whl/nightly/cu128",
        "torch", "torchvision",
        "--force-reinstall", "-q"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    return True


def main():
    cuda_ok, version = check_cuda()

    if cuda_ok and "cu128" in str(version):
        print(f"✅ PyTorch CUDA già configurato: {version}")
    else:
        print(f"Current torch: {version}")
        print("CUDA non disponibile o versione sbagliata, installo nightly...")
        if not install_torch_nightly():
            print("❌ Installazione fallita")
            sys.exit(1)

    # Reload e verifica
    import importlib
    import torch
    importlib.reload(torch)

    print(f"\n{'='*50}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability(0)
        print(f"Compute capability: sm_{cap[0]}{cap[1]}")
        print(f"{'='*50}")
        print("✅ Setup completato!")
    else:
        print(f"{'='*50}")
        print("❌ CUDA non disponibile dopo installazione")
        sys.exit(1)


if __name__ == "__main__":
    main()
