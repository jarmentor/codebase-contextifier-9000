"""Language-specific relationship extractors using strategy pattern.

Each language has its own extractor class that knows how to extract
relationships (CALLS, IMPORTS, EXTENDS, etc.) from AST nodes.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .models import CodeChunk, CodeRelationship

logger = logging.getLogger(__name__)


class RelationshipExtractor(ABC):
    """Base class for language-specific relationship extraction."""

    def __init__(self, language: str):
        self.language = language

    @abstractmethod
    def get_call_node_types(self) -> List[str]:
        """Return AST node types that represent function/method calls."""
        pass

    @abstractmethod
    def get_import_node_types(self) -> List[str]:
        """Return AST node types that represent imports/includes."""
        pass

    @abstractmethod
    def extract_call_target_name(self, call_node: Any) -> Optional[str]:
        """Extract the function/method name being called."""
        pass

    @abstractmethod
    def extract_import_info(self, import_node: Any, source_code: bytes) -> Optional[Dict[str, Any]]:
        """Extract import/require information."""
        pass

    @abstractmethod
    def extract_inheritance_info(self, class_node: Any, source_code: bytes) -> Dict[str, List[str]]:
        """Extract base classes and interfaces from a class node."""
        pass


class PythonExtractor(RelationshipExtractor):
    """Python-specific relationship extraction."""

    def __init__(self):
        super().__init__("python")

    def get_call_node_types(self) -> List[str]:
        return ["call"]

    def get_import_node_types(self) -> List[str]:
        return ["import_statement", "import_from_statement"]

    def extract_call_target_name(self, call_node: Any) -> Optional[str]:
        """Extract function/method name from Python call node."""
        try:
            # call -> function (identifier or attribute)
            func_node = call_node.child_by_field_name("function")
            if func_node:
                if func_node.type == "identifier":
                    return func_node.text.decode("utf-8")
                elif func_node.type == "attribute":
                    # Get the attribute name (last part)
                    attr_node = func_node.child_by_field_name("attribute")
                    if attr_node:
                        return attr_node.text.decode("utf-8")
        except Exception as e:
            logger.debug(f"Error extracting Python call target: {e}")
        return None

    def extract_import_info(self, import_node: Any, source_code: bytes) -> Optional[Dict[str, Any]]:
        """Extract import information from Python import node."""
        try:
            if import_node.type == "import_statement":
                # import module [as alias]
                module_node = import_node.child_by_field_name("name")
                if module_node:
                    module_name = module_node.text.decode("utf-8")
                    return {"module": module_name, "is_relative": False}

            elif import_node.type == "import_from_statement":
                # from module import name
                module_node = import_node.child_by_field_name("module_name")
                if module_node:
                    module_name = module_node.text.decode("utf-8")
                    return {"module": module_name, "is_relative": module_name.startswith(".")}
        except Exception as e:
            logger.debug(f"Error extracting Python import info: {e}")
        return None

    def extract_inheritance_info(self, class_node: Any, source_code: bytes) -> Dict[str, List[str]]:
        """Extract base classes from Python class node."""
        info = {"extends": [], "implements": []}
        try:
            # class ClassName(BaseClass, Interface):
            bases_node = class_node.child_by_field_name("superclasses")
            if bases_node:
                for child in bases_node.children:
                    if child.type == "identifier":
                        info["extends"].append(child.text.decode("utf-8"))
        except Exception as e:
            logger.debug(f"Error extracting Python inheritance: {e}")
        return info


class JavaScriptExtractor(RelationshipExtractor):
    """JavaScript-specific relationship extraction."""

    def __init__(self):
        super().__init__("javascript")

    def get_call_node_types(self) -> List[str]:
        return ["call_expression"]

    def get_import_node_types(self) -> List[str]:
        return ["import_statement"]

    def extract_call_target_name(self, call_node: Any) -> Optional[str]:
        """Extract function/method name from JavaScript call node."""
        try:
            func_node = call_node.child_by_field_name("function")
            if func_node:
                if func_node.type == "identifier":
                    return func_node.text.decode("utf-8")
                elif func_node.type == "member_expression":
                    prop_node = func_node.child_by_field_name("property")
                    if prop_node:
                        return prop_node.text.decode("utf-8")
        except Exception as e:
            logger.debug(f"Error extracting JavaScript call target: {e}")
        return None

    def extract_import_info(self, import_node: Any, source_code: bytes) -> Optional[Dict[str, Any]]:
        """Extract import information from JavaScript import node."""
        try:
            # import ... from "module"
            source_node = import_node.child_by_field_name("source")
            if source_node:
                module_name = source_node.text.decode("utf-8").strip('"\'')
                return {"module": module_name, "is_relative": module_name.startswith(".")}
        except Exception as e:
            logger.debug(f"Error extracting JavaScript import info: {e}")
        return None

    def extract_inheritance_info(self, class_node: Any, source_code: bytes) -> Dict[str, List[str]]:
        """Extract base classes and interfaces from JavaScript class node."""
        info = {"extends": [], "implements": []}
        try:
            # class ClassName extends BaseClass
            heritage_node = class_node.child_by_field_name("heritage")
            if heritage_node:
                for child in heritage_node.children:
                    if child.type == "extends_clause":
                        for subchild in child.children:
                            if subchild.type == "identifier":
                                info["extends"].append(subchild.text.decode("utf-8"))
        except Exception as e:
            logger.debug(f"Error extracting JavaScript inheritance: {e}")
        return info


class TypeScriptExtractor(JavaScriptExtractor):
    """TypeScript-specific relationship extraction (extends JavaScript)."""

    def __init__(self):
        super().__init__()
        self.language = "typescript"

    def extract_inheritance_info(self, class_node: Any, source_code: bytes) -> Dict[str, List[str]]:
        """Extract base classes and interfaces from TypeScript class node."""
        info = {"extends": [], "implements": []}
        try:
            # class ClassName extends BaseClass implements Interface
            heritage_node = class_node.child_by_field_name("heritage")
            if heritage_node:
                for child in heritage_node.children:
                    if child.type == "extends_clause":
                        for subchild in child.children:
                            if subchild.type == "identifier":
                                info["extends"].append(subchild.text.decode("utf-8"))
                    elif child.type == "implements_clause":
                        for subchild in child.children:
                            if subchild.type == "type_identifier":
                                info["implements"].append(subchild.text.decode("utf-8"))
        except Exception as e:
            logger.debug(f"Error extracting TypeScript inheritance: {e}")
        return info


class PHPExtractor(RelationshipExtractor):
    """PHP-specific relationship extraction."""

    def __init__(self):
        super().__init__("php")

    def get_call_node_types(self) -> List[str]:
        return ["function_call_expression", "member_call_expression"]

    def get_import_node_types(self) -> List[str]:
        return ["namespace_use_declaration", "include_expression", "require_expression"]

    def extract_call_target_name(self, call_node: Any) -> Optional[str]:
        """Extract function/method name from PHP call node."""
        try:
            # Handle member_call_expression: $object->method()
            if call_node.type == "member_call_expression":
                name_node = call_node.child_by_field_name("name")
                if name_node:
                    return name_node.text.decode("utf-8")

            # Handle function_call_expression: function_name()
            elif call_node.type == "function_call_expression":
                function_node = call_node.child_by_field_name("function")
                if function_node:
                    # Handle namespaced functions: Namespace\function_name
                    if function_node.type == "qualified_name":
                        # Get just the function name, not the namespace
                        for child in function_node.children:
                            if child.type == "name":
                                return child.text.decode("utf-8")
                    # Handle simple function calls: function_name
                    else:
                        return function_node.text.decode("utf-8")
        except Exception as e:
            logger.debug(f"Error extracting PHP call target: {e}")
        return None

    def extract_import_info(self, import_node: Any, source_code: bytes) -> Optional[Dict[str, Any]]:
        """Extract import information from PHP namespace use node."""
        try:
            if import_node.type == "namespace_use_declaration":
                for child in import_node.children:
                    if child.type == "namespace_name":
                        module_name = child.text.decode("utf-8")
                        return {"module": module_name, "is_relative": False}
        except Exception as e:
            logger.debug(f"Error extracting PHP import info: {e}")
        return None

    def extract_inheritance_info(self, class_node: Any, source_code: bytes) -> Dict[str, List[str]]:
        """Extract base classes and interfaces from PHP class node."""
        info = {"extends": [], "implements": []}
        try:
            # class ClassName extends BaseClass implements Interface
            base_node = class_node.child_by_field_name("base_clause")
            if base_node:
                for child in base_node.children:
                    if child.type == "name":
                        info["extends"].append(child.text.decode("utf-8"))

            interface_node = class_node.child_by_field_name("interface_clause")
            if interface_node:
                for child in interface_node.children:
                    if child.type == "name":
                        info["implements"].append(child.text.decode("utf-8"))
        except Exception as e:
            logger.debug(f"Error extracting PHP inheritance: {e}")
        return info


class GoExtractor(RelationshipExtractor):
    """Go-specific relationship extraction."""

    def __init__(self):
        super().__init__("go")

    def get_call_node_types(self) -> List[str]:
        return ["call_expression"]

    def get_import_node_types(self) -> List[str]:
        return ["import_declaration"]

    def extract_call_target_name(self, call_node: Any) -> Optional[str]:
        """Extract function name from Go call node."""
        try:
            func_node = call_node.child_by_field_name("function")
            if func_node:
                if func_node.type == "identifier":
                    return func_node.text.decode("utf-8")
                elif func_node.type == "selector_expression":
                    field_node = func_node.child_by_field_name("field")
                    if field_node:
                        return field_node.text.decode("utf-8")
        except Exception as e:
            logger.debug(f"Error extracting Go call target: {e}")
        return None

    def extract_import_info(self, import_node: Any, source_code: bytes) -> Optional[Dict[str, Any]]:
        """Extract import information from Go import node."""
        try:
            # import "package" or import ( "package1" "package2" )
            for child in import_node.children:
                if child.type == "import_spec":
                    path_node = child.child_by_field_name("path")
                    if path_node:
                        module_name = path_node.text.decode("utf-8").strip('"')
                        return {"module": module_name, "is_relative": module_name.startswith(".")}
        except Exception as e:
            logger.debug(f"Error extracting Go import info: {e}")
        return None

    def extract_inheritance_info(self, class_node: Any, source_code: bytes) -> Dict[str, List[str]]:
        """Go uses composition, not inheritance."""
        return {"extends": [], "implements": []}


class RustExtractor(RelationshipExtractor):
    """Rust-specific relationship extraction."""

    def __init__(self):
        super().__init__("rust")

    def get_call_node_types(self) -> List[str]:
        return ["call_expression"]

    def get_import_node_types(self) -> List[str]:
        return ["use_declaration"]

    def extract_call_target_name(self, call_node: Any) -> Optional[str]:
        """Extract function name from Rust call node."""
        try:
            func_node = call_node.child_by_field_name("function")
            if func_node:
                if func_node.type == "identifier":
                    return func_node.text.decode("utf-8")
                elif func_node.type == "field_expression":
                    field_node = func_node.child_by_field_name("field")
                    if field_node:
                        return field_node.text.decode("utf-8")
        except Exception as e:
            logger.debug(f"Error extracting Rust call target: {e}")
        return None

    def extract_import_info(self, import_node: Any, source_code: bytes) -> Optional[Dict[str, Any]]:
        """Extract import information from Rust use node."""
        try:
            # use std::collections::HashMap;
            text = import_node.text.decode("utf-8")
            module_name = text.replace("use ", "").replace(";", "").strip()
            return {"module": module_name, "is_relative": False}
        except Exception as e:
            logger.debug(f"Error extracting Rust import info: {e}")
        return None

    def extract_inheritance_info(self, class_node: Any, source_code: bytes) -> Dict[str, List[str]]:
        """Extract trait implementations from Rust struct/impl node."""
        info = {"extends": [], "implements": []}
        # Rust uses traits, would need to parse impl blocks
        return info


class JavaExtractor(RelationshipExtractor):
    """Java-specific relationship extraction."""

    def __init__(self):
        super().__init__("java")

    def get_call_node_types(self) -> List[str]:
        return ["method_invocation"]

    def get_import_node_types(self) -> List[str]:
        return ["import_declaration"]

    def extract_call_target_name(self, call_node: Any) -> Optional[str]:
        """Extract method name from Java call node."""
        try:
            name_node = call_node.child_by_field_name("name")
            if name_node:
                return name_node.text.decode("utf-8")
        except Exception as e:
            logger.debug(f"Error extracting Java call target: {e}")
        return None

    def extract_import_info(self, import_node: Any, source_code: bytes) -> Optional[Dict[str, Any]]:
        """Extract import information from Java import node."""
        try:
            text = import_node.text.decode("utf-8")
            module_name = text.replace("import ", "").replace(";", "").strip()
            return {"module": module_name, "is_relative": False}
        except Exception as e:
            logger.debug(f"Error extracting Java import info: {e}")
        return None

    def extract_inheritance_info(self, class_node: Any, source_code: bytes) -> Dict[str, List[str]]:
        """Extract base classes and interfaces from Java class node."""
        info = {"extends": [], "implements": []}
        try:
            # class ClassName extends BaseClass implements Interface
            superclass_node = class_node.child_by_field_name("superclass")
            if superclass_node:
                for child in superclass_node.children:
                    if child.type == "type_identifier":
                        info["extends"].append(child.text.decode("utf-8"))

            interfaces_node = class_node.child_by_field_name("interfaces")
            if interfaces_node:
                for child in interfaces_node.children:
                    if child.type == "type_identifier":
                        info["implements"].append(child.text.decode("utf-8"))
        except Exception as e:
            logger.debug(f"Error extracting Java inheritance: {e}")
        return info


class CppExtractor(RelationshipExtractor):
    """C++-specific relationship extraction."""

    def __init__(self):
        super().__init__("cpp")

    def get_call_node_types(self) -> List[str]:
        return ["call_expression"]

    def get_import_node_types(self) -> List[str]:
        return ["preproc_include"]

    def extract_call_target_name(self, call_node: Any) -> Optional[str]:
        """Extract function name from C++ call node."""
        try:
            func_node = call_node.child_by_field_name("function")
            if func_node:
                if func_node.type == "identifier":
                    return func_node.text.decode("utf-8")
                elif func_node.type == "field_expression":
                    field_node = func_node.child_by_field_name("field")
                    if field_node:
                        return field_node.text.decode("utf-8")
        except Exception as e:
            logger.debug(f"Error extracting C++ call target: {e}")
        return None

    def extract_import_info(self, import_node: Any, source_code: bytes) -> Optional[Dict[str, Any]]:
        """Extract include information from C++ include node."""
        try:
            # #include <header> or #include "header"
            text = import_node.text.decode("utf-8")
            module_name = text.replace("#include", "").strip().strip('<>"')
            return {"module": module_name, "is_relative": '"' in text}
        except Exception as e:
            logger.debug(f"Error extracting C++ include info: {e}")
        return None

    def extract_inheritance_info(self, class_node: Any, source_code: bytes) -> Dict[str, List[str]]:
        """Extract base classes from C++ class node."""
        info = {"extends": [], "implements": []}
        try:
            # class ClassName : public BaseClass
            base_list_node = class_node.child_by_field_name("base_class_clause")
            if base_list_node:
                for child in base_list_node.children:
                    if child.type == "type_identifier":
                        info["extends"].append(child.text.decode("utf-8"))
        except Exception as e:
            logger.debug(f"Error extracting C++ inheritance: {e}")
        return info


class CExtractor(CppExtractor):
    """C-specific relationship extraction (similar to C++)."""

    def __init__(self):
        super().__init__()
        self.language = "c"

    def extract_inheritance_info(self, class_node: Any, source_code: bytes) -> Dict[str, List[str]]:
        """C doesn't have classes/inheritance."""
        return {"extends": [], "implements": []}


class CSharpExtractor(RelationshipExtractor):
    """C#-specific relationship extraction."""

    def __init__(self):
        super().__init__("c_sharp")

    def get_call_node_types(self) -> List[str]:
        return ["invocation_expression"]

    def get_import_node_types(self) -> List[str]:
        return ["using_directive"]

    def extract_call_target_name(self, call_node: Any) -> Optional[str]:
        """Extract method name from C# call node."""
        try:
            func_node = call_node.child_by_field_name("function")
            if func_node:
                if func_node.type == "identifier":
                    return func_node.text.decode("utf-8")
                elif func_node.type == "member_access_expression":
                    name_node = func_node.child_by_field_name("name")
                    if name_node:
                        return name_node.text.decode("utf-8")
        except Exception as e:
            logger.debug(f"Error extracting C# call target: {e}")
        return None

    def extract_import_info(self, import_node: Any, source_code: bytes) -> Optional[Dict[str, Any]]:
        """Extract using information from C# using node."""
        try:
            # using System.Collections;
            text = import_node.text.decode("utf-8")
            module_name = text.replace("using", "").replace(";", "").strip()
            return {"module": module_name, "is_relative": False}
        except Exception as e:
            logger.debug(f"Error extracting C# using info: {e}")
        return None

    def extract_inheritance_info(self, class_node: Any, source_code: bytes) -> Dict[str, List[str]]:
        """Extract base classes and interfaces from C# class node."""
        info = {"extends": [], "implements": []}
        try:
            # class ClassName : BaseClass, IInterface
            base_list = class_node.child_by_field_name("base_list")
            if base_list:
                # First is usually base class, rest are interfaces
                bases = []
                for child in base_list.children:
                    if child.type == "identifier":
                        bases.append(child.text.decode("utf-8"))

                if bases:
                    # Convention: if name starts with 'I', it's an interface
                    for base in bases:
                        if base.startswith("I") and len(base) > 1 and base[1].isupper():
                            info["implements"].append(base)
                        else:
                            info["extends"].append(base)
        except Exception as e:
            logger.debug(f"Error extracting C# inheritance: {e}")
        return info


class ExtractorRegistry:
    """Registry for language-specific relationship extractors."""

    _extractors: Dict[str, RelationshipExtractor] = {
        "python": PythonExtractor(),
        "javascript": JavaScriptExtractor(),
        "typescript": TypeScriptExtractor(),
        "php": PHPExtractor(),
        "go": GoExtractor(),
        "rust": RustExtractor(),
        "java": JavaExtractor(),
        "cpp": CppExtractor(),
        "c": CExtractor(),
        "c_sharp": CSharpExtractor(),
    }

    @classmethod
    def get_extractor(cls, language: str) -> Optional[RelationshipExtractor]:
        """Get the relationship extractor for a language.

        Args:
            language: Programming language name

        Returns:
            RelationshipExtractor instance or None if not supported
        """
        return cls._extractors.get(language)

    @classmethod
    def register_extractor(cls, language: str, extractor: RelationshipExtractor):
        """Register a custom relationship extractor.

        Args:
            language: Programming language name
            extractor: RelationshipExtractor instance
        """
        cls._extractors[language] = extractor

    @classmethod
    def get_supported_languages(cls) -> List[str]:
        """Get list of languages with relationship extraction support."""
        return list(cls._extractors.keys())
