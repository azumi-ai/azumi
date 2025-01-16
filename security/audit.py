@dataclass
class AuditConfig:
    enabled: bool = True
    log_level: str = "INFO"
    audit_file: str = "audit.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB

class SecurityAuditor:
    def __init__(self, config: Optional[AuditConfig] = None):
        self.config = config or AuditConfig()
        self._setup_logging()
        
    def _setup_logging(self):
        audit_logger = logging.getLogger("security_audit")
        audit_logger.setLevel(self.config.log_level)
        
        handler = logging.handlers.RotatingFileHandler(
            self.config.audit_file,
            maxBytes=self.config.max_file_size,
            backupCount=5
        )
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        audit_logger.addHandler(handler)
        
    async def audit_action(
        self,
        action: str,
        user_id: str,
        details: Dict[str, Any]
    ) -> None:
        if not self.config.enabled:
            return
            
        audit_entry = {
            "timestamp": time.time(),
            "action": action,
            "user_id": user_id,
            "details": details
        }
        
        logging.getLogger("security_audit").info(
            json.dumps(audit_entry)
        )
