# How the Model Registry Decorator Works

## Overview

The decorator `@ModelRegistry.register_language_model("t5")` is a **class decorator** that automatically registers your model class in a dictionary when the module is imported.

## Step-by-Step Breakdown

### 1. The Decorator Definition

```python
@classmethod
def register_language_model(cls, name: str):
    """Decorator to register a language model."""
    def decorator(model_class):
        cls._language_models[name] = model_class
        return model_class
    return decorator
```

This is a **decorator factory** - it returns a decorator function.

### 2. How It Works

When you write:

```python
@ModelRegistry.register_language_model("t5")
class T5LanguageModel(LanguageModel):
    ...
```

Here's what happens:

1. **First call**: `ModelRegistry.register_language_model("t5")` is called with `"t5"` as the name
   - This returns the `decorator` function (with `name="t5"` captured in its closure)

2. **Second call**: Python calls the returned `decorator` function with `T5LanguageModel` as the argument
   - The decorator stores `T5LanguageModel` in `ModelRegistry._language_models["t5"]`
   - Returns the class unchanged (so it can still be used normally)

### 3. Visual Flow

```
@ModelRegistry.register_language_model("t5")
class T5LanguageModel(...):
    pass

# Is equivalent to:

def decorator(model_class):
    ModelRegistry._language_models["t5"] = model_class
    return model_class

T5LanguageModel = decorator(T5LanguageModel)
```

### 4. What Happens at Import Time

When you do:
```python
import audiocaption.models  # This imports t5.py
```

The moment Python imports `t5.py`, it executes the decorator, which:
- Takes the `T5LanguageModel` class
- Stores it in `ModelRegistry._language_models["t5"]`
- The class is now registered and can be retrieved by name!

### 5. Retrieving the Model

Later, when you do:
```python
model = ModelRegistry.get_language_model("t5", config)
```

It:
1. Looks up `"t5"` in `_language_models` dictionary
2. Gets the `T5LanguageModel` class
3. Instantiates it: `T5LanguageModel(config)`
4. Returns the instance

## Complete Example

```python
# Step 1: Define the decorator (in registry.py)
class ModelRegistry:
    _language_models = {}
    
    @classmethod
    def register_language_model(cls, name: str):
        def decorator(model_class):
            cls._language_models[name] = model_class  # Store class
            return model_class  # Return unchanged
        return decorator

# Step 2: Use the decorator (in t5.py)
@ModelRegistry.register_language_model("t5")
class T5LanguageModel:
    def __init__(self, config):
        ...

# Step 3: Import triggers registration
import audiocaption.models  # T5LanguageModel is now registered!

# Step 4: Retrieve by name
model = ModelRegistry.get_language_model("t5", {"model_name": "t5-small"})
# This is equivalent to: T5LanguageModel({"model_name": "t5-small"})
```

## Why This Pattern?

1. **Automatic Registration**: Models register themselves when imported
2. **No Manual Registration**: You don't need to manually add models to a list
3. **Easy Discovery**: Just import the module and the model is available
4. **Clean Code**: The decorator is right where the class is defined

## Alternative (Without Decorator)

Without the decorator, you'd have to do:

```python
# In some central file
ModelRegistry._language_models["t5"] = T5LanguageModel
ModelRegistry._language_models["gpt2"] = GPT2LanguageModel
# ... for every model
```

With the decorator, each model registers itself automatically!
