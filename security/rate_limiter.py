@dataclass
class RateLimitConfig:
    requests_per_minute: int = 60
    burst_size: int = 10
    window_size: int = 60  # seconds

class RateLimiter:
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self.requests = {}
        
    async def check_rate_limit(self, user_id: str) -> bool:
        now = time.time()
        user_requests = self.requests.get(user_id, [])
        
        # Remove old requests
        user_requests = [
            timestamp for timestamp in user_requests
            if timestamp > now - self.config.window_size
        ]
        
        # Check limits
        if len(user_requests) >= self.config.requests_per_minute:
            return False
            
        # Add new request
        user_requests.append(now)
        self.requests[user_id] = user_requests
        return True
        
    async def get_remaining_requests(self, user_id: str) -> int:
        now = time.time()
        user_requests = self.requests.get(user_id, [])
        
        # Count valid requests
        valid_requests = len([
            timestamp for timestamp in user_requests
            if timestamp > now - self.config.window_size
        ])
        
        return max(0, self.config.requests_per_minute - valid_requests)
