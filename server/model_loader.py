import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
from server.logger import get_logger

# Load environment variables from .env file
load_dotenv()
logger = get_logger(__name__)

# Cache for loaded models
_model_cache = {}


def detect_device():
    """
    Detects the best available device (CUDA, MPS, or CPU).
    Returns both the device string and torch device object.
    """
    if torch.cuda.is_available():
        return 'cuda', torch.device('cuda')
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return 'mps', torch.device('mps')
    else:
        return 'cpu', torch.device('cpu')

def load_text_model(model_choice):
    """
    Loads and caches the specified text model.
    """
    global _model_cache
    if model_choice in _model_cache:
        logger.info(f"Model '{model_choice}' loaded from cache.")
        return _model_cache[model_choice]
    logger.info(f"Loading text model '{model_choice}'...")
    device_name, device = detect_device()
    logger.info(f"Using device: {device_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_choice,
        torch_dtype=torch.float16 if device_name != 'cpu' else torch.float32,
        device_map="auto",
        trust_remote_code=True

    )
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_choice, use_fast=True)

    _model_cache[model_choice] = (model, tokenizer, device)
    logger.info(f"Text model '{model_choice}' loaded and cached.")
    return _model_cache[model_choice]

def load_model(model_choice):
    """
    Loads and caches the specified model.
    """
    global _model_cache

    if model_choice in _model_cache:
        logger.info(f"Model '{model_choice}' loaded from cache.")
        return _model_cache[model_choice]

    logger.info(f"Loading vlm model '{model_choice}'...")

    device_name, device = detect_device()
    logger.info(f"Using device: {device_name}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_choice,
        torch_dtype=torch.float16 if device_name != 'cpu' else torch.float32,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_choice)
    model.to(device)
    _model_cache[model_choice] = (model, processor, device)
    logger.info(f"{model_choice} vlm model loaded and cached.")
    return _model_cache[model_choice]