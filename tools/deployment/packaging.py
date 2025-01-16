from typing import Dict, List, Optional, Any
import shutil
import os
import asyncio
from dataclasses import dataclass
import logging
import subprocess

logger = logging.getLogger(__name__)

@dataclass
class PackagingConfig:
    output_dir: str = "dist"
    version_file: str = "version.txt"
    exclude_patterns: List[str] = None
    compression_level: int = 9

class DeploymentPackager:
    """Deployment packaging system."""
    
    def __init__(self, config: PackagingConfig):
        self.config = config
        if not self.config.exclude_patterns:
            self.config.exclude_patterns = [
                "__pycache__",
                "*.pyc",
                "*.pyo",
                "*.pyd",
                ".git",
                ".env",
                "dist",
                "build"
            ]
    
    async def create_package(
        self,
        source_dir: str,
        package_name: str,
        version: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create deployment package."""
        try:
            # Prepare output directory
            os.makedirs(self.config.output_dir, exist_ok=True)
            
            # Create package path
            package_path = os.path.join(
                self.config.output_dir,
                f"{package_name}-{version}.zip"
            )
            
            # Create package
            await self._create_archive(
                source_dir,
                package_path,
                metadata
            )
            
            # Verify package
            if not await self._verify_package(package_path):
                raise ValueError("Package verification failed")
                
            return package_path
            
        except Exception as e:
            logger.error(f"Packaging error: {e}")
            raise
