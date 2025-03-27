import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

class DataManager:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.data_dir / 'data_operations.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def save_metrics(self, metrics: dict, output_dir: str = "output"):
        """
        Save processing metrics to CSV
        """
        try:
            output_path = Path(output_dir) / 'metrics.csv'
            df = pd.DataFrame([metrics])
            df.to_csv(output_path, mode='a', header=not output_path.exists(), index=False)
            self.logger.info(f"Saved metrics to {output_path}")
            return str(output_path)
        except Exception as e:
            self.logger.error(f"Error saving metrics: {str(e)}")
            raise

    def load_metrics(self, output_dir: str = "output") -> pd.DataFrame:
        """
        Load all processing metrics from CSV
        """
        try:
            metrics_path = Path(output_dir) / 'metrics.csv'
            if metrics_path.exists():
                return pd.read_csv(metrics_path)
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error loading metrics: {str(e)}")
            raise

    def export_results(self, output_dir: str = "output", format: str = "csv"):
        """
        Export results in specified format
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(output_dir)
            
            if format == "csv":
                metrics = self.load_metrics(output_dir)
                export_path = output_path / f"canopy_analysis_{timestamp}.csv"
                metrics.to_csv(export_path, index=False)
                self.logger.info(f"Exported results to {export_path}")
                return str(export_path)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            self.logger.error(f"Error exporting results: {str(e)}")
            raise

    def get_analysis_summary(self, output_dir: str = "output") -> dict:
        """
        Generate summary statistics from processed images
        """
        try:
            metrics = self.load_metrics(output_dir)
            if metrics.empty:
                return {}
            
            summary = {
                "total_images": len(metrics),
                "average_canopy_ratio": metrics["canopy_ratio"].mean(),
                "min_canopy_ratio": metrics["canopy_ratio"].min(),
                "max_canopy_ratio": metrics["canopy_ratio"].max(),
                "total_canopy_pixels": metrics["canopy_pixels"].sum(),
                "processing_dates": metrics["timestamp"].unique().tolist()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating summary: {str(e)}")
            raise 