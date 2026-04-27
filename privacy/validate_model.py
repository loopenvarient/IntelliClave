# privacy/validate_model.py
from opacus.validators import ModuleValidator

def validate_m1_model(model):
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        print("❌ Model has Opacus incompatibilities:")
        for e in errors:
            print(f"   {e}")
        print("\nAttempting auto-fix...")
        model = ModuleValidator.fix(model)
        errors_after = ModuleValidator.validate(model, strict=False)
        if errors_after:
            print("❌ Auto-fix failed. Tell M1 to replace BatchNorm with GroupNorm.")
        else:
            print("✅ Auto-fix succeeded. Use this fixed model.")
    else:
        print("✅ Model is fully Opacus-compatible. No changes needed.")
    return model