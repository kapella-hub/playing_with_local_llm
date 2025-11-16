"""Local LLM wrapper using llama-cpp-python."""
import logging
import traceback
from functools import lru_cache
from typing import Optional

from llama_cpp import Llama

from app.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_llm() -> Llama:
    """
    Load and cache the Llama model.
    
    Returns:
        Llama: Initialized Llama model instance
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If model loading fails
    """
    if not settings.validate_model_exists():
        raise FileNotFoundError(
            f"Model file not found at {settings.llm_model_path}. "
            f"Please download a .gguf model and place it in the models/ directory."
        )
    
    logger.info(f"Loading LLM from {settings.llm_model_path}")
    logger.info(f"Configuration: context_size={settings.llm_context_size}, "
                f"n_threads={settings.llm_n_threads}, "
                f"n_gpu_layers={settings.llm_n_gpu_layers}")
    
    try:
        llm = Llama(
            model_path=settings.llm_model_path,
            n_ctx=settings.llm_context_size,
            n_threads=settings.llm_n_threads,
            n_gpu_layers=settings.llm_n_gpu_layers,
            verbose=True  # Enable verbose to capture loading errors
        )
        logger.info("LLM loaded successfully")
        return llm
    except AssertionError as e:
        # Catch AssertionError specifically - usually indicates corrupted/incomplete model file
        error_msg = (
            "Model file appears to be corrupted or incomplete. "
            "Common causes: incomplete download, file transfer error, or incompatible format. "
            "Please re-download the model file."
        )
        logger.error(f"Failed to load LLM: {error_msg}")
        logger.error(f"Original error: {type(e).__name__}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise RuntimeError(error_msg)
    except Exception as e:
        logger.error(f"Failed to load LLM: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise RuntimeError(f"Failed to load LLM: {e}")


def _is_mistral_model() -> bool:
    """
    Detect if the loaded model is a Mistral model based on the model path.
    
    Returns:
        bool: True if Mistral model, False otherwise
    """
    model_path_lower = settings.llm_model_path.lower()
    return "mistral" in model_path_lower


def generate_answer(
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None
) -> str:
    """
    Generate an answer using the local LLM.
    
    Args:
        prompt: User prompt/question
        system_prompt: Optional system prompt for instructions
        max_tokens: Maximum tokens to generate (defaults to settings)
        temperature: Temperature for sampling (defaults to settings)
        
    Returns:
        str: Generated text response
    """
    llm = get_llm()
    
    max_tokens = max_tokens or settings.llm_max_tokens
    temperature = temperature or settings.llm_temperature
    
    # Build chat-style prompt based on model type
    if system_prompt:
        if _is_mistral_model():
            # Mistral 7B Instruct v0.2 format
            # Format: [INST] system_instruction\n\nuser_query [/INST]
            full_prompt = f"[INST] {system_prompt}\n\n{prompt} [/INST]"
            stop_tokens = ["[INST]", "</s>"]
            logger.debug("Using Mistral prompt format")
        else:
            # Llama-style format (default)
            full_prompt = f"""<|system|>
{system_prompt}
<|user|>
{prompt}
<|assistant|>
"""
            stop_tokens = ["<|user|>", "<|system|>"]
            logger.debug("Using Llama prompt format")
    else:
        full_prompt = prompt
        stop_tokens = []
    
    logger.debug(f"Generating answer with max_tokens={max_tokens}, temperature={temperature}")
    
    try:
        response = llm(
            full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop_tokens if stop_tokens else None,
            echo=False
        )
        
        # Extract text from response
        answer = response["choices"][0]["text"].strip()
        logger.debug(f"Generated answer: {answer[:100]}...")
        
        return answer
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        raise RuntimeError(f"Error generating answer: {e}")
