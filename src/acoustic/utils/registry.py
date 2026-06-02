from typing import Any, Callable, Dict, Optional
import inspect


class PluginRegistry:
    """
    A named registry for callable/class components.
    Can optionally enforce a minimal interface by comparing signatures.
    """

    def __init__(self, name: str, interface: Optional[Callable] = None):
        self._name = name
        self._items: Dict[str, Any] = {}
        self._interface = interface

    def register(self, name: str, obj: Any) -> None:
        if name in self._items:
            raise ValueError(f"{self._name} '{name}' is already registered")
        if self._interface is not None:
            self._check_interface(obj)
        self._items[name] = obj

    def get(self, name: str) -> Any:
        if name not in self._items:
            raise KeyError(
                f"Unknown {self._name}: '{name}'. "
                f"Available: {list(self._items.keys())}"
            )
        return self._items[name]

    def _check_interface(self, obj: Any) -> None:
        if not callable(obj):
            raise TypeError(f"{self._name} must be callable")
        try:
            obj_sig = inspect.signature(obj)
            iface_sig = inspect.signature(self._interface)

            if len(obj_sig.parameters) != len(iface_sig.parameters):
                raise TypeError(
                    f"{self._name} signature mismatch: "
                    f"expected {iface_sig}, got {obj_sig}"
                )
        except Exception:
            pass

    def __contains__(self, name: str) -> bool:
        return name in self._items