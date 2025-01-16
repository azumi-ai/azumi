from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging
import json
import csv
import os
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ExportConfig:
    output_dir: str = "reports"
    supported_formats: List[str] = None
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    compression: bool = True

class ReportExporter:
    """Analytics report export system."""
    
    def __init__(self, config: ExportConfig):
        self.config = config
        if not self.config.supported_formats:
            self.config.supported_formats = [
                'json', 'csv', 'excel', 'html', 'pdf', 'markdown'
            ]
        self._ensure_output_dir()
    
    def _ensure_output_dir(self) -> None:
        """Ensure output directory exists."""
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    async def export_report(
        self,
        report: Dict[str, Any],
        format: str,
        filename: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Export report to file."""
        if format not in self.config.supported_formats:
            raise ValueError(f"Unsupported format: {format}")
            
        # Generate filename if not provided
        if not filename:
            filename = self._generate_filename(format)
            
        filepath = Path(self.config.output_dir) / filename
        
        try:
            # Convert report to specified format
            content = await self._convert_format(report, format)
            
            # Check file size
            if len(content.encode('utf-8')) > self.config.max_file_size:
                raise ValueError("Report exceeds maximum file size")
            
            # Save file
            await self._save_file(content, filepath, format)
            
            # Compress if enabled and applicable
            if self.config.compression and format in ['json', 'csv', 'txt']:
                filepath = await self._compress_file(filepath)
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Export error: {e}")
            raise
    
    async def _convert_format(
        self,
        report: Dict[str, Any],
        format: str
    ) -> Union[str, bytes]:
        """Convert report to specified format."""
        if format == 'json':
            return json.dumps(report, indent=2)
        elif format == 'csv':
            return self._to_csv(report)
        elif format == 'excel':
            return self._to_excel(report)
        elif format == 'html':
            return self._to_html(report)
        elif format == 'pdf':
            return await self._to_pdf(report)
        elif format == 'markdown':
            return self._to_markdown(report)
        else:
            raise ValueError(f"Conversion not implemented for format: {format}")
    
    async def _save_file(
        self,
        content: Union[str, bytes],
        filepath: Path,
        format: str
    ) -> None:
        """Save content to file."""
        mode = 'wb' if isinstance(content, bytes) else 'w'
        encoding = None if isinstance(content, bytes) else 'utf-8'
        
        with open(filepath, mode, encoding=encoding) as f:
            f.write(content)
    
    async def _compress_file(self, filepath: Path) -> Path:
        """Compress file using gzip."""
        import gzip
        import shutil
        
        compressed_path = filepath.with_suffix(filepath.suffix + '.gz')
        with open(filepath, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                
        os.remove(filepath)  # Remove original file
        return compressed_path
    
    def _generate_filename(self, format: str) -> str:
        """Generate unique filename for report."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"report_{timestamp}.{format}"
