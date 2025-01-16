from typing import Dict, List, Optional, Any
import jinja2
from dataclasses import dataclass
import logging
import json
import yaml

logger = logging.getLogger(__name__)

@dataclass
class TemplateConfig:
    template_dir: str = "templates"
    custom_filters: Dict[str, callable] = None
    default_format: str = "markdown"
    cache_templates: bool = True

class ReportTemplates:
    """Report template management system."""
    
    def __init__(self, config: TemplateConfig):
        self.config = config
        self.env = self._setup_jinja()
        self.templates = {}
        self.loaded_templates = {}
    
    def _setup_jinja(self) -> jinja2.Environment:
        """Setup Jinja2 environment with custom filters."""
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.config.template_dir),
            autoescape=True,
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Add default filters
        env.filters.update({
            'format_number': self._format_number,
            'format_date': self._format_date,
            'format_duration': self._format_duration,
            'highlight_value': self._highlight_value
        })
        
        # Add custom filters
        if self.config.custom_filters:
            env.filters.update(self.config.custom_filters)
            
        return env
    
    async def load_template(
        self,
        template_name: str,
        format: Optional[str] = None
    ) -> jinja2.Template:
        """Load report template."""
        format = format or self.config.default_format
        template_key = f"{template_name}.{format}"
        
        # Check cache
        if template_key in self.loaded_templates and self.config.cache_templates:
            return self.loaded_templates[template_key]
        
        try:
            template = self.env.get_template(template_key)
            
            if self.config.cache_templates:
                self.loaded_templates[template_key] = template
                
            return template
            
        except Exception as e:
            logger.error(f"Template loading error: {e}")
            raise
