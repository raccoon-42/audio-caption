"""
Simple example demonstrating how the decorator works.
Run this to see the decorator in action!
"""

# Simulate the registry
class SimpleRegistry:
    _models = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator factory - returns a decorator function."""
        print(f"Step 1: register() called with name='{name}'")
        
        def decorator(model_class):
            print(f"Step 2: decorator() called with class={model_class.__name__}")
            cls._models[name] = model_class
            print(f"Step 3: Stored {model_class.__name__} as '{name}'")
            print(f"       Registry now has: {list(cls._models.keys())}")
            return model_class  # Return the class unchanged
        
        return decorator
    
    @classmethod
    def get(cls, name: str):
        """Get a model by name."""
        return cls._models.get(name)


# Using the decorator
print("=" * 50)
print("DEFINING CLASS WITH DECORATOR")
print("=" * 50)

@SimpleRegistry.register("my_model")
class MyModel:
    def __init__(self, config):
        self.config = config
        print(f"MyModel initialized with {config}")


print("\n" + "=" * 50)
print("RETRIEVING MODEL BY NAME")
print("=" * 50)

# Now we can get it by name!
ModelClass = SimpleRegistry.get("my_model")
print(f"Retrieved: {ModelClass}")

# Instantiate it
instance = ModelClass({"param": "value"})
print(f"Instance: {instance}")


print("\n" + "=" * 50)
print("WHAT HAPPENED?")
print("=" * 50)
print("""
When Python sees:
    @SimpleRegistry.register("my_model")
    class MyModel:
        ...

It does:
    1. Call SimpleRegistry.register("my_model")
       → Returns the decorator function
    
    2. Call decorator(MyModel)
       → Stores MyModel in _models["my_model"]
       → Returns MyModel (unchanged)
    
    3. MyModel is now registered and can be retrieved by name!
""")
