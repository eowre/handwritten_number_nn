import pandas as pd
import re 
from datetime import datetime

class LogHandler:
    def __init__(self):
        """
        Initializes the LogHandler.
        This class is responsible for parsing log files and extracting relevant information.
        """
        pass

    def parse(self, log_file):
        """
        Parses the log file and extracts relevant information into a DataFrame.
        """
        records = []
        patteren = re.compile(
            r'(?P<timestamp>[\d\-]+\s[\d:,]+)\s-\s'
            r'Epoch\s(?P<epoch>\d+)/(?P<total_epochs>\d+)\s\|\s'
            r'Processed:\s(?P<processed>\d+)/(?P<total_images>\d+)\s\|\s'
            r'Average Loss:\s(?P<avg_loss>[\d.]+)\s\|\s'
            r'Accuracy:\s(?P<accuracy>[\d.]+)%\s\|\s'
            r'Images Left:\s(?P<images_left>\d+)\s\|\s'
            r'Batch Time:\s(?P<batch_time>[\d.]+)s'
        )

        with open(log_file, 'r') as file:
            for line in file:
                match = patteren.search(line)
                if match:
                    record = match.groupdict()
                    record['timestamp'] = datetime.strptime(record['timestamp'], '%Y-%m-%d %H:%M:%S,%f')
                    record['epoch'] = int(record['epoch'])
                    record['total_epochs'] = int(record['total_epochs'])
                    record['processed'] = int(record['processed'])
                    record['total_images'] = int(record['total_images'])
                    record['avg_loss'] = float(record['avg_loss'])
                    record['accuracy'] = float(record['accuracy'])
                    record['images_left'] = int(record['images_left'])
                    record['batch_time'] = float(record['batch_time'])
                    records.append(record)
        return pd.DataFrame(records) 
    
    def extract_activations(self, log_file):
        """
        Extracts layer information from the log file.
        """
        # Remove the 'model_' prefix
        raw = log_file.replace("model_", "").replace(".log", "")
        # Split by underscores
        parts = raw.split("_")

        sizes = [int(p) for p in parts if p.isdigit()]
        activations = [p for p in parts if not p.isdigit()]

        return   " → ".join(
            f"{size}({act})" for size, act in zip(sizes, activations + [""])
        )