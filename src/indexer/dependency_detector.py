"""Dependency detection for WordPress, Composer, and npm packages."""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DependencyDetector:
    """Detect dependencies in a workspace (WordPress plugins, Composer, npm)."""

    def __init__(self, workspace_path: str):
        """Initialize dependency detector.

        Args:
            workspace_path: Path to the workspace/project root
        """
        self.workspace_path = Path(workspace_path)

    def detect_all(self) -> Dict[str, List[Dict[str, Any]]]:
        """Detect all available dependencies in the workspace.

        Returns:
            Dictionary with keys:
            - wordpress_plugins: List of WordPress plugin metadata
            - composer_packages: List of Composer package metadata
            - npm_packages: List of npm package metadata
        """
        return {
            "wordpress_plugins": self.detect_wordpress_plugins(),
            "wordpress_themes": self.detect_wordpress_themes(),
            "composer_packages": self.detect_composer_packages(),
            "npm_packages": self.detect_npm_packages(),
        }

    def detect_wordpress_plugins(self) -> List[Dict[str, Any]]:
        """Detect WordPress plugins in wp-content/plugins.

        Returns:
            List of plugin metadata dicts
        """
        plugins = []
        plugins_dir = self.workspace_path / "wp-content" / "plugins"

        if not plugins_dir.exists():
            logger.debug("No wp-content/plugins directory found")
            return plugins

        for plugin_dir in plugins_dir.iterdir():
            if not plugin_dir.is_dir():
                continue

            # Skip common non-plugin directories
            if plugin_dir.name.startswith("."):
                continue

            # Try to find main plugin file (usually plugin-name.php)
            plugin_file = None
            for php_file in plugin_dir.glob("*.php"):
                if self._is_plugin_file(php_file):
                    plugin_file = php_file
                    break

            if plugin_file:
                metadata = self._parse_plugin_header(plugin_file)
                if metadata:
                    metadata["path"] = str(plugin_dir)
                    metadata["slug"] = plugin_dir.name
                    plugins.append(metadata)
                    logger.debug(f"Detected plugin: {metadata['name']} v{metadata['version']}")

        return plugins

    def detect_wordpress_themes(self) -> List[Dict[str, Any]]:
        """Detect WordPress themes in wp-content/themes.

        Returns:
            List of theme metadata dicts
        """
        themes = []
        themes_dir = self.workspace_path / "wp-content" / "themes"

        if not themes_dir.exists():
            logger.debug("No wp-content/themes directory found")
            return themes

        for theme_dir in themes_dir.iterdir():
            if not theme_dir.is_dir() or theme_dir.name.startswith("."):
                continue

            # Look for style.css with theme header
            style_css = theme_dir / "style.css"
            if style_css.exists():
                metadata = self._parse_theme_header(style_css)
                if metadata:
                    metadata["path"] = str(theme_dir)
                    metadata["slug"] = theme_dir.name
                    themes.append(metadata)
                    logger.debug(f"Detected theme: {metadata['name']} v{metadata['version']}")

        return themes

    def detect_composer_packages(self) -> List[Dict[str, Any]]:
        """Detect Composer packages from composer.lock.

        Returns:
            List of package metadata dicts
        """
        packages = []
        composer_lock = self.workspace_path / "composer.lock"

        if not composer_lock.exists():
            logger.debug("No composer.lock found")
            return packages

        try:
            lock_data = json.loads(composer_lock.read_text())

            for package in lock_data.get("packages", []):
                vendor_path = self.workspace_path / "vendor" / package["name"]

                if vendor_path.exists():
                    packages.append({
                        "name": package["name"],
                        "version": package.get("version", "unknown"),
                        "description": package.get("description", ""),
                        "path": str(vendor_path),
                        "type": "composer_package",
                    })
                    logger.debug(f"Detected composer package: {package['name']} v{package.get('version')}")

        except Exception as e:
            logger.error(f"Error parsing composer.lock: {e}")

        return packages

    def detect_npm_packages(self) -> List[Dict[str, Any]]:
        """Detect npm packages from package.json (top-level dependencies only).

        Note: Does NOT include all transitive dependencies - only direct deps.
        User should explicitly choose which packages to index.

        Returns:
            List of package metadata dicts
        """
        packages = []
        package_json = self.workspace_path / "package.json"

        if not package_json.exists():
            logger.debug("No package.json found")
            return packages

        try:
            pkg_data = json.loads(package_json.read_text())
            dependencies = {
                **pkg_data.get("dependencies", {}),
                **pkg_data.get("devDependencies", {}),
            }

            for name, version_spec in dependencies.items():
                node_modules_path = self.workspace_path / "node_modules" / name

                if node_modules_path.exists():
                    # Try to read version from node_modules/package/package.json
                    pkg_json = node_modules_path / "package.json"
                    actual_version = version_spec
                    description = ""

                    if pkg_json.exists():
                        try:
                            pkg_info = json.loads(pkg_json.read_text())
                            actual_version = pkg_info.get("version", version_spec)
                            description = pkg_info.get("description", "")
                        except Exception:
                            pass

                    packages.append({
                        "name": name,
                        "version": actual_version,
                        "description": description,
                        "path": str(node_modules_path),
                        "type": "npm_package",
                    })
                    logger.debug(f"Detected npm package: {name} v{actual_version}")

        except Exception as e:
            logger.error(f"Error parsing package.json: {e}")

        return packages

    def _is_plugin_file(self, php_file: Path) -> bool:
        """Check if a PHP file contains WordPress plugin headers.

        Args:
            php_file: Path to PHP file

        Returns:
            True if file has plugin headers
        """
        try:
            content = php_file.read_text(encoding="utf-8", errors="ignore")
            # Look for "Plugin Name:" in first 8KB
            return "Plugin Name:" in content[:8192]
        except Exception:
            return False

    def _parse_plugin_header(self, plugin_file: Path) -> Optional[Dict[str, str]]:
        """Parse WordPress plugin header.

        Args:
            plugin_file: Path to main plugin PHP file

        Returns:
            Dict with name, version, description, author, etc.
        """
        try:
            content = plugin_file.read_text(encoding="utf-8", errors="ignore")
            headers = {}

            # Parse plugin header fields
            fields = {
                "Plugin Name": "name",
                "Version": "version",
                "Description": "description",
                "Author": "author",
                "Author URI": "author_uri",
                "Plugin URI": "plugin_uri",
                "Text Domain": "text_domain",
            }

            for wp_field, key in fields.items():
                pattern = rf"{re.escape(wp_field)}:\s*(.+?)(?:\n|$)"
                match = re.search(pattern, content[:8192], re.IGNORECASE)
                if match:
                    headers[key] = match.group(1).strip()

            if "name" in headers and "version" in headers:
                headers["type"] = "wordpress_plugin"
                return headers

        except Exception as e:
            logger.error(f"Error parsing plugin header: {e}")

        return None

    def _parse_theme_header(self, style_css: Path) -> Optional[Dict[str, str]]:
        """Parse WordPress theme header from style.css.

        Args:
            style_css: Path to theme style.css

        Returns:
            Dict with name, version, description, etc.
        """
        try:
            content = style_css.read_text(encoding="utf-8", errors="ignore")
            headers = {}

            # Parse theme header fields
            fields = {
                "Theme Name": "name",
                "Version": "version",
                "Description": "description",
                "Author": "author",
                "Author URI": "author_uri",
                "Theme URI": "theme_uri",
                "Text Domain": "text_domain",
            }

            for wp_field, key in fields.items():
                pattern = rf"{re.escape(wp_field)}:\s*(.+?)(?:\n|$)"
                match = re.search(pattern, content[:8192], re.IGNORECASE)
                if match:
                    headers[key] = match.group(1).strip()

            if "name" in headers:
                headers["type"] = "wordpress_theme"
                headers.setdefault("version", "unknown")
                return headers

        except Exception as e:
            logger.error(f"Error parsing theme header: {e}")

        return None

    def get_dependency_by_name(
        self, name: str, dep_type: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific dependency by name.

        Args:
            name: Dependency name (plugin slug, package name, etc.)
            dep_type: Optional type filter (wordpress_plugin, composer_package, npm_package)

        Returns:
            Dependency metadata dict or None if not found
        """
        all_deps = self.detect_all()

        for category, deps in all_deps.items():
            if dep_type and category != dep_type + "s":
                continue

            for dep in deps:
                # Match by name or slug
                if dep.get("name") == name or dep.get("slug") == name:
                    return dep

        return None
