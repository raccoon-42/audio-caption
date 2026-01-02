# Decorator Execution - Simple Explanation

## The Key Point

**Python automatically calls the decorator when it sees the `@` symbol.** You don't call it yourself!

## What Python Does Automatically

When you write:

```python
@ModelRegistry.register_language_model("t5")
class T5LanguageModel:
    pass
```

Python **automatically transforms** it into this:

```python
class T5LanguageModel:
    pass

# Python automatically adds this line:
T5LanguageModel = ModelRegistry.register_language_model("t5")(T5LanguageModel)
```

## Step-by-Step: What Happens When Python Reads Your Code

Let's trace through exactly what happens:

### Step 1: Python sees the decorator

```python
@ModelRegistry.register_language_model("t5")
```

Python thinks: "I need to call this function and use its result as a decorator"

### Step 2: Python calls `register_language_model("t5")`

```python
# Python does this automatically:
decorator_function = ModelRegistry.register_language_model("t5")
```

This executes:
```python
def register_language_model(cls, name: str):  # name = "t5"
    def decorator(model_class):  # This function is created
        cls._language_models[name] = model_class
        return model_class
    return decorator  # Returns the decorator function
```

So `decorator_function` is now the inner `decorator` function (with `name="t5"` remembered).

### Step 3: Python defines the class

```python
class T5LanguageModel:
    pass
```

The class object is created.

### Step 4: Python automatically calls the decorator

```python
# Python automatically does this:
T5LanguageModel = decorator_function(T5LanguageModel)
```

This executes:
```python
def decorator(model_class):  # model_class = T5LanguageModel
    cls._language_models["t5"] = T5LanguageModel  # Store it!
    return T5LanguageModel  # Return it unchanged
```

## Visual Timeline

```
Time 0: Python reads your file
  ↓
Time 1: Python sees @ModelRegistry.register_language_model("t5")
  ↓
Time 2: Python calls register_language_model("t5")
        → Returns decorator function
  ↓
Time 3: Python defines class T5LanguageModel
  ↓
Time 4: Python automatically calls decorator(T5LanguageModel)
        → Stores T5LanguageModel in registry
        → Returns T5LanguageModel
  ↓
Time 5: T5LanguageModel is now registered!
```

## You Don't Call Anything!

The important thing: **You never call the decorator yourself!**

- ❌ You don't write: `ModelRegistry.register_language_model("t5")(T5LanguageModel)`
- ✅ You just write: `@ModelRegistry.register_language_model("t5")` above the class
- ✅ Python does the rest automatically!

## Complete Example with Print Statements

Here's what happens if we add print statements:

```python
class ModelRegistry:
    _models = {}
    
    @classmethod
    def register(cls, name: str):
        print(f"1. register() called with name='{name}'")
        
        def decorator(model_class):
            print(f"2. decorator() called with class={model_class.__name__}")
            cls._models[name] = model_class
            print(f"3. Stored {model_class.__name__} as '{name}'")
            return model_class
        
        print(f"4. Returning decorator function")
        return decorator

# When Python reads this file, it will print:
# 1. register() called with name='t5'
# 4. Returning decorator function
# 2. decorator() called with class=T5LanguageModel
# 3. Stored T5LanguageModel as 't5'

@ModelRegistry.register("t5")
class T5LanguageModel:
    pass

# All of the above happens automatically when Python
# reads this file - you don't need to do anything!
```

## The "Second Call" Confusion

When I said "second call", I meant:

1. **First**: Python calls `register_language_model("t5")` to get the decorator
2. **Second**: Python automatically calls the returned `decorator(T5LanguageModel)`

But you don't make these calls - **Python does it automatically** when it processes the `@` decorator syntax!

## Summary

- The `@` symbol tells Python: "call this function and use its result"
- Python automatically calls the decorator function with your class
- This happens when Python reads the file, not when you import it
- You never need to manually call the decorator - it's all automatic!
