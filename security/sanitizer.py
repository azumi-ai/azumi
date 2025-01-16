class InputSanitizer:
    @staticmethod
    def sanitize_text(text: str) -> str:
        # Remove potentially harmful characters
        sanitized = re.sub(r'[^\w\s\-.,!?]', '', text)
        return sanitized.strip()
        
    @staticmethod
    def sanitize_json(data: Dict[str, Any]) -> Dict[str, Any]:
        # Recursively sanitize JSON data
        def sanitize_value(value):
            if isinstance(value, str):
                return InputSanitizer.sanitize_text(value)
            elif isinstance(value, dict):
                return {k: sanitize_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [sanitize_value(v) for v in value]
            return value
            
        return sanitize_value(data)
        
    @staticmethod
    def validate_file_type(filename: str, allowed_types: List[str]) -> bool:
        extension = filename.lower().split('.')[-1]
        return extension in allowed_types

# Initialize security components with default configurations
monitor = SecurityMonitor()
auditor = SecurityAuditor()
rate_limiter = RateLimiter()
sanitizer = InputSanitizer()

# Example usage in a middleware
async def security_middleware(request, call_next):
    user_id = request.headers.get("X-User-Id")
    
    # Check rate limit
    if not await rate_limiter.check_rate_limit(user_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Sanitize input
    if request.json:
        request.json = sanitizer.sanitize_json(request.json)
    
    # Monitor and audit
    await monitor.log_security_event({
        "type": "request",
        "severity": "info",
        "session_id": user_id
    })
    
    await auditor.audit_action(
        action="api_request",
        user_id=user_id,
        details={"path": request.url.path}
    )
    
    # Process request
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    # Update metrics
    monitor.response_time.observe(duration)
    monitor.requests_total.inc()
    
    return response
