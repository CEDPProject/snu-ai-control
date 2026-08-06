import logging
import logging.handlers
import os
import sys

class ColoredFormatter(logging.Formatter):
    """컬러와 레벨별 포맷을 지원하는 포매터"""
    
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    
    FORMATS = {
        logging.DEBUG: grey + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.INFO: "%(asctime)s - %(name)s - %(message)s",  # name 추가
        logging.WARNING: yellow + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.ERROR: red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.CRITICAL: bold_red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset
    }
    
    def format(self, record):
        # 모듈 이름을 짧게 만들기
        record.name = record.name.split('.')[-1]
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%H:%M:%S')
        return formatter.format(record)

def setup_default_logging(default_level=logging.INFO, log_path='', silent=False):
    """개선된 로깅 설정
    
    Args:
        default_level: 로그 레벨
        log_path: 로그 파일 경로
        silent: True면 콘솔 출력 비활성화 (파일에만 로깅)
    """
    # 기존 핸들러 모두 제거
    logging.root.handlers.clear()
    
    # 콘솔 핸들러 (silent가 False일 때만 추가)
    if not silent:
        # Keep logs and unavoidable third-party prints on the same stream.
        # Mixing stderr logging with buffered stdout made preprocessing
        # messages appear after training had already finished.
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setFormatter(ColoredFormatter())
        logging.root.addHandler(console_handler)
    
    # 파일 핸들러 (항상 추가)
    if log_path:
        # 디렉토리가 없으면 생성
        log_dir = os.path.dirname(log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, 
            maxBytes=(1024 ** 2 * 2), 
            backupCount=3,
            encoding='utf-8',
        )
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)20s: [%(levelname)8s] - %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logging.root.addHandler(file_handler)
    
    logging.root.setLevel(default_level)
        
# 로깅 헬퍼 클래스 추가
class TrainingLogger:
    """Accelerator와 통합된 로깅 클래스"""
    def __init__(self, accelerator, name=None):
        self.accelerator = accelerator
        self.logger = logging.getLogger(name or __name__)
        
        # 파일 전용 로거 설정
        self.file_logger = logging.getLogger(f"{name}.file_only")
        self.file_logger.propagate = False
        self.file_logger.setLevel(logging.INFO)
        
        # 루트 로거의 파일 핸들러만 복사
        for handler in logging.root.handlers:
            if isinstance(handler, (logging.FileHandler, logging.handlers.RotatingFileHandler)):
                self.file_logger.addHandler(handler)
    
    def info(self, message, *args, **kwargs):
        """Log INFO on the main process using the standard Logger contract."""
        if self.accelerator.is_main_process:
            self.logger.info(message, *args, **kwargs)

    def info_file_only(self, message, *args, **kwargs):
        """Log INFO only to the run file on the main process."""
        if self.accelerator.is_main_process:
            self.file_logger.info(message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        """Log WARNING with process context."""
        prefix = f"[Process {self.accelerator.process_index}] "
        self.logger.warning(prefix + str(message), *args, **kwargs)

    def error(self, message, *args, **kwargs):
        """Log ERROR with process context."""
        prefix = f"[Process {self.accelerator.process_index}] "
        self.logger.error(prefix + str(message), *args, **kwargs)

    def debug(self, message, *args, **kwargs):
        """Log DEBUG when DEBUG_MODE is enabled."""
        if os.environ.get('DEBUG_MODE', 'false').lower() == 'true':
            prefix = f"[Process {self.accelerator.process_index}] "
            self.logger.debug(prefix + str(message), *args, **kwargs)

def setup_main_logging(accelerator, savedir, debug=False, silent=False):
    """학습용 로깅 설정 - setup_default_logging 활용
    
    Args:
        accelerator: Accelerator 인스턴스
        savedir: 저장 디렉토리 (예: '/path/to/experiments/SMAP/version1')
        debug: 디버그 모드 활성화
        silent: True면 터미널 출력 비활성화 (파일에만 로깅)
    """
    
    # 로그 레벨 결정
    if debug:
        level = logging.DEBUG
    elif accelerator.is_main_process:
        level = logging.INFO
    else:
        level = logging.WARNING  # 다른 프로세스는 WARNING만
    
    # 로그 파일 경로 (메인 프로세스만)
    log_path = ''
    if accelerator.is_main_process:
        # savedir가 없으면 생성
        if not os.path.exists(savedir):
            os.makedirs(savedir, exist_ok=True)
        
        # savedir에서 프로세스 이름 추출 (마지막 디렉토리명)
        process_name = os.path.basename(os.path.normpath(savedir))
        
        # 로그 파일은 savedir 내부에 저장
        # 예: savedir = '/path/to/experiments/SMAP/version1'
        #     log_path = '/path/to/experiments/SMAP/version1/version1.log'
        log_path = os.path.join(savedir, f"{process_name}.log")
    
    # 기존 핸들러 제거 (중복 방지)
    logging.root.handlers.clear()
    
    # setup_default_logging 호출 (silent 파라미터 전달)
    setup_default_logging(default_level=level, log_path=log_path, silent=silent)
    
    # 특정 로거들의 레벨 조정 (노이즈 감소)
    if not debug:
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('accelerate').setLevel(logging.WARNING)
        logging.getLogger('transformers').setLevel(logging.WARNING)
    
    return TrainingLogger(accelerator, 'MAIN')