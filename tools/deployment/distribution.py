from typing import Dict, List, Optional, Any
import aiohttp
import asyncio
from dataclasses import dataclass
import logging
import os

logger = logging.getLogger(__name__)

@dataclass
class DistributionConfig:
    chunk_size: int = 8192
    concurrent_uploads: int = 3
    retry_attempts: int = 3
    timeout: float = 300  # seconds

class DeploymentDistributor:
    """Deployment distribution system."""
    
    def __init__(self, config: DistributionConfig):
        self.config = config
        self.session = None
        self._semaphore = asyncio.Semaphore(config.concurrent_uploads)
    
    async def __aenter__(self):
        """Initialize distributor."""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup distributor."""
        if self.session:
            await self.session.close()
    
    async def distribute(
        self,
        package_path: str,
        targets: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Distribute deployment package."""
        if not os.path.exists(package_path):
            raise FileNotFoundError(f"Package not found: {package_path}")
            
        results = {}
        tasks = []
        
        # Create upload tasks
        for target in targets:
            task = self._upload_to_target(
                package_path,
                target,
                metadata
            )
            tasks.append(task)
        
        # Execute uploads
        async with self._semaphore:
            results = await asyncio.gather(
                *tasks,
                return_exceptions=True
            )
        
        return {
            'package': package_path,
            'targets': len(targets),
            'successful': sum(1 for r in results if not isinstance(r, Exception)),
            'failed': sum(1 for r in results if isinstance(r, Exception)),
            'results': results
        }
    
    async def _upload_to_target(
        self,
        package_path: str,
        target: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Upload package to target."""
        for attempt in range(self.config.retry_attempts):
            try:
                async with self.session.post(
                    target['url'],
                    data=self._create_upload_data(package_path, metadata),
                    headers=target.get('headers', {}),
                    timeout=self.config.timeout
                ) as response:
                    response.raise_for_status()
                    return {
                        'target': target['url'],
                        'status': response.status,
                        'attempt': attempt + 1
                    }
            except Exception as e:
                if attempt == self.config.retry_attempts - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
