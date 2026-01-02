"""
Test to show when the decorator actually runs.
Run this and see the print statements!
"""

print("=" * 60)
print("BEFORE defining the class")
print("=" * 60)


class ModelRegistry:
    _models = {}
    
    @classmethod
    def register(cls, name: str):
        print(f"\n[STEP 1] register() called with name='{name}'")
        print(f"         This happens when Python sees @ModelRegistry.register('{name}')")
        
        def decorator(model_class):
            print(f"\n[STEP 2] decorator() called with class={model_class.__name__}")
            print(f"         This happens automatically when Python processes the class")
            cls._models[name] = model_class
            print(f"[STEP 3] Stored {model_class.__name__} in registry as '{name}'")
            return model_class
        
        print(f"[STEP 1.5] Returning decorator function")
        return decorator


print("\n" + "=" * 60)
print("DEFINING CLASS - Watch the print statements!")
print("=" * 60)

@ModelRegistry.register("t5")
class T5LanguageModel:
    """This class is being defined right now."""
    pass

print("\n" + "=" * 60)
print("AFTER defining the class")
print("=" * 60)
print(f"Registry now contains: {list(ModelRegistry._models.keys())}")
print(f"T5LanguageModel class: {T5LanguageModel}")

print("\n" + "=" * 60)
print("KEY INSIGHT")
print("=" * 60)
print("""
All those print statements happened AUTOMATICALLY when Python
read the @ModelRegistry.register("t5") line!

You didn't call anything - Python did it for you when it processed
the decorator syntax.
""")
