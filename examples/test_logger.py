"""
Example script demonstrating how to import and use the logger
"""
from banana_net.utils.logger import logger

if __name__ == "__main__":
    logger.info("This is a test log message from banana_net")
    logger.debug("Debug message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    print("Logging test completed. Check logs directory for output file.")