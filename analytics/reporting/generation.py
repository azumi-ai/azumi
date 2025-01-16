from typing import Dict, List, Optional, Any
import asyncio
from dataclasses import dataclass
import logging
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    max_concurrent: int = 5
    timeout: float = 300  # seconds
    chunk_size: int = 1000
    include_metadata: bool = True

class ReportGenerator:
    """Analytics report generation system."""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.templates = ReportTemplates()
        self._semaphore = asyncio.Semaphore(config.max_concurrent)
    
    async def generate_report(
        self,
        data: Dict[str, Any],
        template_name: str,
        format: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate analytics report."""
        async with self._semaphore:
            try:
                # Load template
                template = await self.templates.load_template(
                    template_name,
                    format
                )
                
                # Process data
                processed_data = await self._process_data(data)
                
                # Generate report content
                content = await self._generate_content(
                    template,
                    processed_data,
                    parameters
                )
                
                # Add metadata if enabled
                if self.config.include_metadata:
                    metadata = self._generate_metadata(
                        template_name,
                        format,
                        parameters
                    )
                    content = self._add_metadata(content, metadata)
                
                return {
                    'content': content,
                    'format': format or self.templates.config.default_format,
                    'timestamp': datetime.now().isoformat(),
                    'template': template_name
                }
                
            except Exception as e:
                logger.error(f"Report generation error: {e}")
                raise
