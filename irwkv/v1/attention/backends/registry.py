# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention backend registry."""

from collections.abc import Callable
from enum import Enum, EnumMeta
from typing import TYPE_CHECKING, cast

from irwkv.logger import init_logger
from irwkv.utils.import_utils import resolve_obj_by_qualname

if TYPE_CHECKING:
    from irwkv.v1.attention.backend import AttentionBackend

logger = init_logger(__name__)


class _AttentionBackendEnumMeta(EnumMeta):
    """Metaclass for AttentionBackendEnum to provide better error messages."""

    def __getitem__(cls, name: str):
        """Get backend by name with helpful error messages."""
        try:
            return super().__getitem__(name)
        except KeyError:
            members = cast("dict[str, Enum]", cls.__members__).keys()
            valid_backends = ", ".join(members)
            raise ValueError(
                f"Unknown attention backend: '{name}'. "
                f"Valid options are: {valid_backends}"
            ) from None


class AttentionBackendEnum(Enum, metaclass=_AttentionBackendEnumMeta):
    """Enumeration of all supported attention backends.

    The enum value is the default class path, but this can be overridden
    at runtime using register_backend().

    To get the actual backend class (respecting overrides), use:
        backend.get_class()
    """

    TORCH_SDPA = ""
    CUSTOM = None

    @classmethod
    def register(cls, name: str, value: str) -> "AttentionBackendEnum":
        """Dynamically register a new backend enum member.

        Args:
            name: The name for the new enum member
            value: The fully qualified class path string

        Returns:
            The newly created enum member

        """
        if name in cls._member_map_:
            raise ValueError(
                f"Backend {name} already exists in {cls.__name__}. "
                f"Use register_backend({cls.__name__}.{name}, '{value}') "
                f"to override."
            )
        if not name.isidentifier() or hasattr(cls, name):
            raise ValueError(f"Invalid or reserved backend name: {name}")

        # Create new enum member dynamically
        member = object.__new__(cls)
        member._name_ = name
        member._value_ = value
        setattr(cls, name, member)
        cls._member_map_[name] = member
        cls._value2member_map_[value] = member
        cls._member_names_.append(name)
        logger.info("Registered new attention backend: %s -> %s", name, value)
        return member

    def get_path(self, include_classname: bool = True) -> str:
        """Get the class path for this backend (respects overrides).

        Returns:
            The fully qualified class path string

        Raises:
            ValueError: If Backend.CUSTOM is used without being registered

        """
        path = _ATTN_OVERRIDES.get(self, self.value)
        if not path:
            raise ValueError(
                f"Backend {self.name} must be registered before use. "
                f"Use register_backend(Backend.{self.name}, 'your.module.YourClass')"
            )
        if not include_classname:
            path = path.rsplit(".", 1)[0]
        return path

    def get_class(self) -> "type[AttentionBackend]":
        """Get the backend class (respects overrides).

        Returns:
            The backend class

        Raises:
            ImportError: If the backend class cannot be imported
            ValueError: If Backend.CUSTOM is used without being registered

        """
        return resolve_obj_by_qualname(self.get_path())

    def is_overridden(self) -> bool:
        """Check if this backend has been overridden.

        Returns:
            True if the backend has a registered override

        """
        return self in _ATTN_OVERRIDES

    def clear_override(self) -> None:
        """Clear any override for this backend, reverting to the default."""
        _ATTN_OVERRIDES.pop(self, None)


class MambaAttentionBackendEnum(Enum, metaclass=_AttentionBackendEnumMeta):
    """Enumeration of all supported mamba attention backends.

    The enum value is the default class path, but this can be overridden
    at runtime using register_backend().

    To get the actual backend class (respecting overrides), use:
        backend.get_class()
    """

    @classmethod
    def register(cls, name: str, value: str) -> "MambaAttentionBackendEnum":
        """Dynamically register a new mamba backend enum member.

        Args:
            name: The name for the new enum member
            value: The fully qualified class path string

        Returns:
            The newly created enum member

        """
        if name in cls._member_map_:
            raise ValueError(
                f"Backend {name} already exists in {cls.__name__}. "
                f"Use register_backend({cls.__name__}.{name}, '{value}') "
                f"to override."
            )
        if not name.isidentifier() or hasattr(cls, name):
            raise ValueError(f"Invalid or reserved backend name: {name}")

        member = object.__new__(cls)
        member._name_ = name
        member._value_ = value
        setattr(cls, name, member)
        cls._member_map_[name] = member
        cls._value2member_map_[value] = member
        cls._member_names_.append(name)
        logger.info("Registered new attention backend: %s -> %s", name, value)
        return member

    RWKV = "irwkv.v1.attention.backends.rwkv_attn.RwkvAttentionBackend"
    # Placeholder for third-party/custom backends - must be registered before use
    # set to None to avoid alias with other backend, whose value is an empty string
    CUSTOM = None

    def get_path(self, include_classname: bool = True) -> str:
        """Get the class path for this backend (respects overrides).

        Returns:
            The fully qualified class path string

        Raises:
            ValueError: If Backend.CUSTOM is used without being registered

        """
        path = _MAMBA_ATTN_OVERRIDES.get(self, self.value)
        if not path:
            raise ValueError(
                f"Backend {self.name} must be registered before use. "
                f"Use register_backend(Backend.{self.name}, 'your.module.YourClass')"
            )
        if not include_classname:
            path = path.rsplit(".", 1)[0]
        return path

    def get_class(self) -> "type[AttentionBackend]":
        """Get the backend class (respects overrides).

        Returns:
            The backend class

        Raises:
            ImportError: If the backend class cannot be imported
            ValueError: If Backend.CUSTOM is used without being registered

        """
        return resolve_obj_by_qualname(self.get_path())

    def is_overridden(self) -> bool:
        """Check if this backend has been overridden.

        Returns:
            True if the backend has a registered override

        """
        return self in _MAMBA_ATTN_OVERRIDES

    def clear_override(self) -> None:
        """Clear any override for this backend, reverting to the default."""
        _MAMBA_ATTN_OVERRIDES.pop(self, None)


_ATTN_OVERRIDES: dict[AttentionBackendEnum, str] = {}
_MAMBA_ATTN_OVERRIDES: dict[MambaAttentionBackendEnum, str] = {}


def register_backend(
    backend: AttentionBackendEnum | MambaAttentionBackendEnum | str,
    class_path: str | None = None,
    is_mamba: bool = False,
) -> Callable[[type], type]:
    """Register or override a backend implementation.

    Args:
        backend: The AttentionBackendEnum/MambaAttentionBackendEnum member to register,
                 or a string name for a new custom backend (e.g., "CUSTOM_MLA").
        class_path: Optional class path. If not provided and used as
            decorator, will be auto-generated from the class.
        is_mamba: Whether the backend is a Mamba attention backend.

    Returns:
        Decorator function if class_path is None, otherwise a no-op

    Examples:
        # Register an attention backend
        @register_backend("MY_ATTENTION")
        class MyAttention:
            ...

        # Override an existing mamba attention backend
        @register_backend("MY_LINEAR", is_mamba=True)
        class MyCustomMambaAttn:
            ...

        # Register a custom third-party attention backend
        @register_backend(AttentionBackendEnum.CUSTOM)
        class MyCustomBackend:
            ...

        # Register a new custom attention backend with a dynamic enum name
        @register_backend("CUSTOM_MLA")
        class CustomMLABackend:
            ...

        # Direct registration
        register_backend(
            AttentionBackendEnum.CUSTOM,
            "my.module.MyCustomBackend"
        )

        # Direct registration with string name
        register_backend(
            "CUSTOM_MLA",
            "custom.attention.mla_v1.CustomMLAABackend"
        )

    """
    # Handle dynamic enum creation for string backend names
    if isinstance(backend, str):
        if is_mamba:
            backend = MambaAttentionBackendEnum.register(backend, class_path or "")
        else:
            backend = AttentionBackendEnum.register(backend, class_path or "")

    def decorator(cls: type) -> type:
        if is_mamba:
            _MAMBA_ATTN_OVERRIDES[backend] = f"{cls.__module__}.{cls.__qualname__}"  # type: ignore[index]
        else:
            _ATTN_OVERRIDES[backend] = f"{cls.__module__}.{cls.__qualname__}"  # type: ignore[index]
        return cls

    if class_path is not None:
        if is_mamba:
            _MAMBA_ATTN_OVERRIDES[backend] = class_path  # type: ignore[index]
        else:
            _ATTN_OVERRIDES[backend] = class_path  # type: ignore[index]
        return lambda x: x

    return decorator
