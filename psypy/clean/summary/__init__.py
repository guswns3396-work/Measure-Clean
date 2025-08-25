import os
import importlib

# Dynamically import all modules in the subpackage directory
for module_name in os.listdir(os.path.dirname(__file__)):
    if module_name.endswith(".py") and module_name != "__init__.py":
        module_name = module_name[:-3]  # Remove the ".py" extension
        importlib.import_module(f".{module_name}", package="psypy.clean.summary")
