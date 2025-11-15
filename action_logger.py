import logging
import os
from datetime import datetime, timezone, timedelta

class ActionLogger:

    def __init__(self, log_file='logs/action_logs.txt'):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        self.logger = logging.getLogger('action_logger')
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            # Set timezone to IST (UTC+5:30)
            ist_timezone = timezone(timedelta(hours=5, minutes=30))
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            formatter.converter = lambda *args: datetime.now(ist_timezone).timetuple()
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def log_action(self, action_type, details=None, user_id=None, result=None):
        message = f"ACTION: {action_type}"
        if user_id:
            message += f" | USER: {user_id}"
        if details:
            message += f" | DETAILS: {details}"
        if result:
            message += f" | RESULT: {result}"

        self.logger.info(message)

    def log_route_access(self, route, method, params=None):
        details = f"Route: {route} | Method: {method}"
        if params:
            details += f" | Params: {params}"
        self.log_action("ROUTE_ACCESS", details)

    def log_data_operation(self, operation, entity_type, entity_id=None, details=None):
        action_details = f"Operation: {operation} | Entity: {entity_type}"
        if entity_id:
            action_details += f" | ID: {entity_id}"
        if details:
            action_details += f" | {details}"
        self.log_action("DATA_OPERATION", action_details)

    def log_ml_operation(self, operation, model_name=None, node_id=None, details=None):
        action_details = f"Operation: {operation}"
        if model_name:
            action_details += f" | Model: {model_name}"
        if node_id:
            action_details += f" | Node: {node_id}"
        if details:
            action_details += f" | {details}"
        self.log_action("ML_OPERATION", action_details)

    def log_error(self, operation, error_message):
        """Log errors"""
        self.log_action("ERROR", f"Operation: {operation} | Error: {error_message}")

action_logger = ActionLogger()
