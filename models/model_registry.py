from .composer_bge_m3 import ComposerBGEM3

COMPOSER_MODEL_REGISTRY = {
    # Original LLM models
    'mosaic_llama_125m': 'llmshearing.models.composer_llama.ComposerMosaicLlama',
    'mosaic_llama_370m': 'llmshearing.models.composer_llama.ComposerMosaicLlama',
    'mosaic_llama_1.3b': 'llmshearing.models.composer_llama.ComposerMosaicLlama',
    'mosaic_llama_3b': 'llmshearing.models.composer_llama.ComposerMosaicLlama',
    'mosaic_llama_7b': 'llmshearing.models.composer_llama.ComposerMosaicLlama',
    'mosaic_llama_13b': 'llmshearing.models.composer_llama.ComposerMosaicLlama',
    'mosaic_llama_30b': 'llmshearing.models.composer_llama.ComposerMosaicLlama',
    'mosaic_llama_65b': 'llmshearing.models.composer_llama.ComposerMosaicLlama',
    'mosaic_pythia_70m': 'llmshearing.models.composer_pythia.ComposerMosaicPythia',
    'mosaic_pythia_160m': 'llmshearing.models.composer_pythia.ComposerMosaicPythia',
    'mosaic_pythia_410m': 'llmshearing.models.composer_pythia.ComposerMosaicPythia',
    'mosaic_pythia_1.4b': 'llmshearing.models.composer_pythia.ComposerMosaicPythia',
    'mosaic_llama2_370m': 'llmshearing.models.composer_llama.ComposerMosaicLlama',
    'mosaic_llama2_1.3b': 'llmshearing.models.composer_llama.ComposerMosaicLlama',
    'mosaic_llama2_3b': 'llmshearing.models.composer_llama.ComposerMosaicLlama',
    'mosaic_llama2_7b': 'llmshearing.models.composer_llama.ComposerMosaicLlama',
    'mosaic_llama2_13b': 'llmshearing.models.composer_llama.ComposerMosaicLlama',
    'mosaic_together_3b': 'llmshearing.models.composer_pythia.ComposerMosaicPythia',
    
    # BGE-M3 embedding models
    'bge_m3_base': ComposerBGEM3,
    'bge_m3_75pct': ComposerBGEM3,
    'bge_m3_60pct': ComposerBGEM3,
    'bge_m3_50pct': ComposerBGEM3,
}

def get_model_class(model_name: str):
    """Get model class from registry"""
    if model_name in COMPOSER_MODEL_REGISTRY:
        model_class = COMPOSER_MODEL_REGISTRY[model_name]
        if isinstance(model_class, str):
            # Import dynamically for backward compatibility
            module_path, class_name = model_class.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            return getattr(module, class_name)
        return model_class
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def register_model(name: str, model_class):
    """Register a new model class"""
    COMPOSER_MODEL_REGISTRY[name] = model_class
