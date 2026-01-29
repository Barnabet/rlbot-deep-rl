"""Component registry for dynamic component instantiation."""

from typing import Any, Callable, Dict, Optional, Type, TypeVar

T = TypeVar("T")


class ComponentRegistry:
    """Registry for dynamically creating components by name."""

    def __init__(self):
        self._registries: Dict[str, Dict[str, Type]] = {}

    def register(self, category: str, name: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register a component class.

        Args:
            category: Category of the component (e.g., 'reward', 'obs_builder')
            name: Name to register the component under

        Returns:
            Decorator function
        """
        if category not in self._registries:
            self._registries[category] = {}

        def decorator(cls: Type[T]) -> Type[T]:
            self._registries[category][name] = cls
            return cls

        return decorator

    def get(self, category: str, name: str) -> Optional[Type]:
        """Get a registered component class.

        Args:
            category: Category of the component
            name: Name of the component

        Returns:
            The registered class or None if not found
        """
        return self._registries.get(category, {}).get(name)

    def create(self, category: str, name: str, *args, **kwargs) -> Any:
        """Create an instance of a registered component.

        Args:
            category: Category of the component
            name: Name of the component
            *args: Positional arguments for the constructor
            **kwargs: Keyword arguments for the constructor

        Returns:
            Instance of the component

        Raises:
            KeyError: If component is not found
        """
        cls = self.get(category, name)
        if cls is None:
            available = list(self._registries.get(category, {}).keys())
            raise KeyError(
                f"Component '{name}' not found in category '{category}'. "
                f"Available: {available}"
            )
        return cls(*args, **kwargs)

    def list_category(self, category: str) -> list:
        """List all registered components in a category.

        Args:
            category: Category to list

        Returns:
            List of component names
        """
        return list(self._registries.get(category, {}).keys())

    def list_categories(self) -> list:
        """List all registered categories.

        Returns:
            List of category names
        """
        return list(self._registries.keys())


# Global registry instance
registry = ComponentRegistry()
