import logging


class ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[95m",  # Purple
        "INFO": "\033[92m",  # Green
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "CRITICAL": "\033[91m" + "\033[1m",  # Bold Red
        "RESET": "\033[0m",
    }

    def format(self, record):
        log_message = super(ColorFormatter, self).format(record)
        log_message = self.COLORS.get(record.levelname, "") + log_message + self.COLORS["RESET"]
        return log_message


my_logger = logging.getLogger("test")
console_handler = logging.StreamHandler()
console_handler.setFormatter(ColorFormatter("%(levelname)s - %(message)s"))
my_logger.addHandler(console_handler)
my_logger.warning("****完成第零次测试****")

logging.basicConfig(level="DEBUG", format="%(message)s")
root_logger = logging.getLogger()
root_logger.warning("****完成第一次测试****")


my_logger.warning("****完成第二次测试****")
